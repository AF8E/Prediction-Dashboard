'use client'

import { useState, useCallback, useMemo, useEffect } from 'react'
import plantumlEncoder from 'plantuml-encoder'
import { useDropzone } from 'react-dropzone'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface SystemPrediction {
  year: number
  fci: number
  state: string
}

interface MarkovMatrix {
  states: string[]
  matrix: number[][]
  degradation_rate_per_year: number
  years_to_failure: number
}

interface PredictionResult {
  success: boolean
  systems: Record<string, number>
  facilities?: Record<string, any>
  projects_by_year: Record<string, any[]>
  markov_matrices: Record<string, MarkovMatrix>
  system_importance?: Record<string, number>
  system_correlations?: Record<string, Array<{ system: string; score: number }>>
  failure_predictions?: Record<string, { failure_year?: number; remaining_years?: number; current_fci: number; degradation_rate: number }>
  system_descriptions?: Record<string, { good: string; fair: string; poor: string; impact: string; importance?: number; failure_year?: number; remaining_years?: number }>
  plantuml_diagram?: string
  plantuml_svg?: string
  stats: {
    projects_rows: number
    inventory_rows: number
    project_years: number
    systems_count: number
    facilities_count?: number
    avg_life_expectancy: number
  }
  results: {
    unfunded: Array<{ year: number; fci: number }>
    funded: Array<{ year: number; fci: number }>
    system_predictions_unfunded: Record<string, SystemPrediction[]>
    system_predictions_funded: Record<string, SystemPrediction[]>
    stop_reason: string
    years_simulated: number
    start_year: number
  }
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

const SYSTEM_COLORS = [
  '#22d3ee', '#f97316', '#a855f7', '#22c55e', '#ef4444',
  '#3b82f6', '#eab308', '#ec4899', '#14b8a6', '#8b5cf6',
  '#f59e0b', '#06b6d4', '#84cc16', '#f43f5e', '#6366f1',
  '#10b981', '#f472b6', '#0ea5e9', '#a3e635', '#fb923c',
]

export default function Home() {
  const [projectsFile, setProjectsFile] = useState<File | null>(null)
  const [inventoryFile, setInventoryFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [appPort, setAppPort] = useState<string>('detecting...')
  const [selectedSystem, setSelectedSystem] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'facility' | 'individual'>('individual')
  const [showMarkovMode, setShowMarkovMode] = useState<'table' | 'diagram'>('diagram')

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setAppPort(window.location.port || '3000')
    }
  }, [])

  const onDropProjects = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setProjectsFile(acceptedFiles[0])
      setError(null)
    }
  }, [])

  const onDropInventory = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setInventoryFile(acceptedFiles[0])
      setError(null)
    }
  }, [])

  const { getRootProps: getProjectsRootProps, getInputProps: getProjectsInputProps, isDragActive: isProjectsDragActive } =
    useDropzone({
      onDrop: onDropProjects,
      accept: {
        'text/csv': ['.csv'],
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
        'application/vnd.ms-excel': ['.xls'],
      },
      multiple: false,
    })

  const { getRootProps: getInventoryRootProps, getInputProps: getInventoryInputProps, isDragActive: isInventoryDragActive } =
    useDropzone({
      onDrop: onDropInventory,
      accept: {
        'text/csv': ['.csv'],
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
        'application/vnd.ms-excel': ['.xls'],
      },
      multiple: false,
    })

  const handlePredict = async () => {
    if (!projectsFile || !inventoryFile) {
      setError('Please upload both files before running prediction.')
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('projects_file', projectsFile)
    formData.append('inventory_file', inventoryFile)

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to process files')
      }

      const data: PredictionResult = await response.json()
      setPredictionResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while processing files')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setProjectsFile(null)
    setInventoryFile(null)
    setPredictionResult(null)
    setError(null)
  }

  const facilityChartData =
    predictionResult?.results.unfunded.map((unfunded, index) => ({
      year: unfunded.year,
      'Unfunded Scenario': unfunded.fci,
      'Funded Scenario': predictionResult.results.funded[index].fci,
    })) || []

  const aggregateMarkovMatrix = useMemo((): MarkovMatrix | null => {
    if (!predictionResult?.markov_matrices) return null
    const matrices = Object.values(predictionResult.markov_matrices)
    if (matrices.length === 0) return null
    const sum = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]
    let rateSum = 0
    let lifeSum = 0
    matrices.forEach((m) => {
      rateSum += m.degradation_rate_per_year
      lifeSum += m.years_to_failure
      m.matrix.forEach((row, r) => {
        row.forEach((val, c) => {
          sum[r][c] += val
        })
      })
    })
    const n = matrices.length
    const avg = sum.map((row) => row.map((v) => +(v / n).toFixed(3)))
    return {
      states: ['Good', 'Fair', 'Poor'],
      matrix: avg,
      degradation_rate_per_year: +(rateSum / n).toFixed(2),
      years_to_failure: Math.round(lifeSum / n),
    }
  }, [predictionResult])

  const renderPlantUml = useCallback(async (plantumlText: string): Promise<string | null> => {
    try {
      const response = await fetch(`${API_BASE}/render-plantuml`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ diagram_source: plantumlText })
      })
      if (response.ok) {
        const data = await response.json()
        if (data.success && data.svg) {
          return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(data.svg)))}`
        }
      }
      const encoded = plantumlEncoder.encode(plantumlText)
      return `https://www.plantuml.com/plantuml/svg/${encoded}`
    } catch (e) {
      console.error('PlantUML rendering error:', e)
      try {
        const encoded = plantumlEncoder.encode(plantumlText)
        return `https://www.plantuml.com/plantuml/svg/${encoded}`
      } catch (fallbackError) {
        console.error('Fallback encoding error:', fallbackError)
        return null
      }
    }
  }, [])

  const [megaPlantUmlUrlFromDiagram, setMegaPlantUmlUrlFromDiagram] = useState<string | null>(null)
  
  useEffect(() => {
    if (predictionResult?.plantuml_diagram) {
      renderPlantUml(predictionResult.plantuml_diagram).then(setMegaPlantUmlUrlFromDiagram)
    } else {
      setMegaPlantUmlUrlFromDiagram(null)
    }
  }, [predictionResult?.plantuml_diagram, renderPlantUml])

  const megaPlantUmlSvg = useMemo(() => predictionResult?.plantuml_svg || null, [predictionResult])
  
  const megaPlantUmlUrl = useMemo(() => {
    if (megaPlantUmlSvg) {
      return null
    }
    return megaPlantUmlUrlFromDiagram
  }, [megaPlantUmlSvg, megaPlantUmlUrlFromDiagram])

  const allSystemsChartData = useMemo(() => {
    if (!predictionResult?.results.system_predictions_unfunded) return []
    
    const systems = Object.keys(predictionResult.results.system_predictions_unfunded)
    if (systems.length === 0) return []
    
    const firstSystem = systems[0]
    const yearCount = predictionResult.results.system_predictions_unfunded[firstSystem]?.length || 0
    
    return Array.from({ length: yearCount }, (_, index) => {
      const dataPoint: Record<string, number> = {
        year: predictionResult.results.system_predictions_unfunded[firstSystem][index]?.year || 0,
      }
      
      systems.forEach((system) => {
        const unfunded = predictionResult.results.system_predictions_unfunded[system]?.[index]
        if (unfunded) {
          dataPoint[`${system}`] = unfunded.fci
        }
      })
      
      return dataPoint
    })
  }, [predictionResult])

  const selectedSystemChartData = useMemo(() => {
    if (!selectedSystem || !predictionResult?.results.system_predictions_unfunded) return []
    
    const unfundedData = predictionResult.results.system_predictions_unfunded[selectedSystem]
    const fundedData = predictionResult.results.system_predictions_funded[selectedSystem]
    
    if (!unfundedData || !fundedData) return []
    
    return unfundedData.map((unfunded, index) => ({
      year: unfunded.year,
      'Unfunded': unfunded.fci,
      'Funded': fundedData[index]?.fci || 0,
    }))
  }, [selectedSystem, predictionResult])

  const chartData = facilityChartData

  const getSystemDescription = useCallback((systemName: string) => {
    if (predictionResult?.system_descriptions?.[systemName]) {
      const apiDesc = predictionResult.system_descriptions[systemName]
      return {
        good: apiDesc.good || '',
        fair: apiDesc.fair || '',
        poor: apiDesc.poor || '',
        impact: apiDesc.impact || ''
      }
    }
    
    const systemLower = systemName.toLowerCase()
    
    const descriptions: Record<string, { good: string; fair: string; poor: string; impact: string }> = {
      foundation: {
        good: "Structurally sound, no settlement or cracks. Supports building loads without issues.",
        fair: "Minor settlement or hairline cracks. Requires monitoring but still functional.",
        poor: "Significant settlement, visible cracks, or structural movement. Risk of building instability.",
        impact: "Foundation failure can cause structural damage to entire building, affecting all systems above."
      },
      basement: {
        good: "Waterproof, no leaks, proper drainage. Fully usable space.",
        fair: "Occasional minor leaks or moisture. Some areas may need attention.",
        poor: "Persistent leaks, water intrusion, or structural issues. Unusable or hazardous.",
        impact: "Basement failure can damage utilities, cause mold, and compromise building structure."
      },
      superstructure: {
        good: "Load-bearing elements intact. No structural concerns. Meets all safety standards.",
        fair: "Some wear or minor issues. May need repairs but still safe and functional.",
        poor: "Significant structural degradation. Safety concerns. May require evacuation.",
        impact: "Superstructure failure can cause building collapse, affecting all occupants and systems."
      },
      "exterior structure": {
        good: "Weather-tight, no water intrusion. Proper insulation and protection.",
        fair: "Minor leaks or wear. Some areas need maintenance but functional.",
        poor: "Water intrusion, deterioration, or failure. Interior damage likely.",
        impact: "Exterior failure allows weather intrusion, damaging interior systems and finishes."
      },
      roofing: {
        good: "Waterproof, no leaks. Proper drainage and insulation. Expected lifespan remaining.",
        fair: "Minor leaks or wear. Some repairs needed but functional.",
        poor: "Persistent leaks, significant deterioration, or failure. Interior damage occurring.",
        impact: "Roof failure causes water damage to all systems below, including structure and interiors."
      },
      hvac: {
        good: "Full capacity operation. Efficient climate control. All zones functional.",
        fair: "Reduced capacity or efficiency. Some zones may have issues. Higher energy costs.",
        poor: "System failure or severe degradation. Inadequate climate control. Occupant discomfort or safety risk.",
        impact: "HVAC failure affects occupant comfort, equipment operation, and can cause mold/moisture issues."
      },
      electric: {
        good: "Full capacity, reliable power. All circuits functional. Meets code requirements.",
        fair: "Some circuits unreliable or overloaded. May need upgrades. Occasional outages.",
        poor: "Frequent outages, safety hazards, or insufficient capacity. Critical systems at risk.",
        impact: "Electric failure shuts down all powered systems: HVAC, lighting, equipment, security."
      },
      plumbing: {
        good: "No leaks, proper pressure, all fixtures functional. Water quality meets standards.",
        fair: "Minor leaks or pressure issues. Some fixtures need repair but mostly functional.",
        poor: "Major leaks, low pressure, or contamination. Water damage or health hazards.",
        impact: "Plumbing failure causes water damage, health risks, and building closure."
      },
      "fire protection": {
        good: "All systems operational. Sprinklers, alarms, and suppression ready. Meets code.",
        fair: "Some systems need maintenance. Minor issues but still functional.",
        poor: "System failures or code violations. Life safety risk. Building may be unoccupiable.",
        impact: "Fire protection failure creates life safety hazard and may violate occupancy permits."
      }
    }
    
    for (const [key, desc] of Object.entries(descriptions)) {
      if (systemLower.includes(key)) {
        return desc
      }
    }
    
    return {
      good: `${systemName} is in excellent condition, fully functional, and meets all operational requirements.`,
      fair: `${systemName} shows signs of wear but remains functional. Some maintenance or repairs needed.`,
      poor: `${systemName} has significant degradation or failures. Operational risk or failure likely.`,
      impact: `${systemName} failure affects facility operations and may impact related systems.`
    }
  }, [predictionResult])

  const [systemDiagramCache, setSystemDiagramCache] = useState<Record<string, string>>({})

  const buildPlantUmlDiagram = useCallback((system: string, matrix: MarkovMatrix, opts?: { title?: string }): string => {
    const pct = (p: number) => Math.max(0, Math.min(100, p * 100)).toFixed(1)
    const title = opts?.title ?? `${system} Markov Transitions`
    const desc = getSystemDescription(system)
    const degRate = matrix.degradation_rate_per_year?.toFixed(2) || '0.00'
    const lifeExp = matrix.years_to_failure || 0
    
    const failurePred = predictionResult?.failure_predictions?.[system]
    const failureInfo = failurePred?.failure_year 
      ? ` | Predicted Failure: ${failurePred.failure_year}`
      : failurePred?.remaining_years 
        ? ` | Remaining: ${failurePred.remaining_years.toFixed(1)}yr`
        : ''
    
    const importance = predictionResult?.system_importance?.[system]
    const importanceText = importance ? ` | Importance: ${importance.toFixed(0)}%` : ''
    
    const escapePlantUml = (text: string) => {
      return text
        .replace(/"/g, "'")  // Replace double quotes with single
        .replace(/\n/g, ' ')  // Replace newlines with spaces
        .replace(/\s+/g, ' ')  // Collapse multiple spaces
        .trim()
    }
    
    const goodDesc = escapePlantUml(desc.good)
    const fairDesc = escapePlantUml(desc.fair)
    const poorDesc = escapePlantUml(desc.poor)
    const impactDesc = escapePlantUml(desc.impact)
    
    const diagram = [
      '@startuml',
      'skinparam backgroundColor #0f172a',
      'skinparam shadowing false',
      'skinparam handwritten false',
      'skinparam ArrowColor white',
      'skinparam ArrowThickness 2',
      'skinparam defaultFontName Inter',
      'skinparam defaultFontColor white',
      'skinparam state {',
      '  BackgroundColor #0b1224',
      '  BorderColor #3b82f6',
      '  FontColor white',
      '}',
      'left to right direction',
      `title ${title}`,
      '',
      `state "Good (67-100% FCI)${failureInfo}" as Good`,
      `state "Fair (34-66% FCI)${failureInfo}" as Fair`,
      `state "Poor (0-33% FCI)${failureInfo}" as Poor`,
      '',
      `Good --> Good : Stay ${pct(matrix.matrix[0][0])}%`,
      `Good --> Fair : Degrade ${pct(matrix.matrix[0][1])}%`,
      `Fair --> Good : Improve ${pct(matrix.matrix[1][0])}%`,
      `Fair --> Fair : Stay ${pct(matrix.matrix[1][1])}%`,
      `Fair --> Poor : Degrade ${pct(matrix.matrix[1][2])}%`,
      `Poor --> Fair : Improve ${pct(matrix.matrix[2][1])}%`,
      `Poor --> Poor : Stay ${pct(matrix.matrix[2][2])}%`,
      '',
      'note right of Good',
      'FCI (Facility Condition Index): 0-100% scale',
      `Good (67-100%): ${goodDesc}`,
      `Fair (34-66%): ${fairDesc}`,
      `Poor (0-33%): ${poorDesc}`,
      `Impact: ${impactDesc}`,
      `Life: ${lifeExp}yr | Degrade: ${degRate}%/yr${importanceText}`,
      failurePred?.failure_year ? `Predicted Failure: Year ${failurePred.failure_year}` : '',
      failurePred?.remaining_years ? `Remaining Life: ${failurePred.remaining_years.toFixed(1)} years` : '',
      'end note',
      '@enduml'
    ].join('\n')
    
    return diagram
  }, [getSystemDescription, predictionResult])

  useEffect(() => {
    if (!predictionResult?.markov_matrices) return
    
    const renderDiagrams = async () => {
      const newCache: Record<string, string> = {}
      for (const [system, matrix] of Object.entries(predictionResult.markov_matrices)) {
        const cacheKey = system
        const diagramText = buildPlantUmlDiagram(system, matrix)
        const svgUrl = await renderPlantUml(diagramText)
        if (svgUrl) {
          newCache[cacheKey] = svgUrl
        }
      }
      setSystemDiagramCache(newCache)
    }
    
    renderDiagrams()
  }, [predictionResult?.markov_matrices, buildPlantUmlDiagram, renderPlantUml])

  useEffect(() => {
    if (!aggregateMarkovMatrix) return
    
    const renderAggregate = async () => {
      const diagramText = buildPlantUmlDiagram('All Components', aggregateMarkovMatrix, { title: 'All Components (Average)' })
      const svgUrl = await renderPlantUml(diagramText)
      if (svgUrl) {
        setSystemDiagramCache(prev => ({ ...prev, 'All Components': svgUrl }))
      }
    }
    
    renderAggregate()
  }, [aggregateMarkovMatrix, buildPlantUmlDiagram, renderPlantUml])

  const buildPlantUmlUrl = useCallback((system: string, matrix: MarkovMatrix, opts?: { title?: string }): string | null => {
    const cacheKey = opts?.title === 'All Components (Average)' ? 'All Components' : system
    return systemDiagramCache[cacheKey] || null
  }, [systemDiagramCache])

  const summary = useMemo(() => {
    if (!predictionResult) return null
    const fundedStart = predictionResult.results.funded[0]?.fci ?? 0
    const fundedEnd = predictionResult.results.funded.at(-1)?.fci ?? 0
    const unfundedEnd = predictionResult.results.unfunded.at(-1)?.fci ?? 0
    const lift = fundedEnd - unfundedEnd
    const avgUnfunded =
      predictionResult.results.unfunded.reduce((acc, v) => acc + v.fci, 0) / predictionResult.results.unfunded.length
    const avgFunded =
      predictionResult.results.funded.reduce((acc, v) => acc + v.fci, 0) / predictionResult.results.funded.length
    return {
      fundedStart: fundedStart.toFixed(1),
      fundedEnd: fundedEnd.toFixed(1),
      lift: lift.toFixed(1),
      avgFunded: avgFunded.toFixed(1),
      avgUnfunded: avgUnfunded.toFixed(1),
    }
  }, [predictionResult])

  const renderUploadCard = (
    title: string,
    description: string,
    file: File | null,
    getRootProps: ReturnType<typeof useDropzone>['getRootProps'],
    getInputProps: ReturnType<typeof useDropzone>['getInputProps'],
    isDragActive: boolean
  ) => (
    <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm backdrop-blur transition hover:-translate-y-0.5 hover:shadow-md">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{description}</p>
          <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
        </div>
        <div className="text-xs text-slate-500">CSV or XLSX</div>
      </div>
      <div
        {...getRootProps()}
        className={`group flex min-h-[150px] flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-6 text-center transition ${
          isDragActive
            ? 'border-blue-500 bg-blue-50/60'
            : file
            ? 'border-emerald-500 bg-emerald-50/60'
            : 'border-slate-300 hover:border-slate-400'
        }`}
      >
        <input {...getInputProps()} />
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-900 text-white shadow-lg shadow-slate-900/10">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="stroke-current">
            <path d="M12 16V4m0 0 4 4m-4-4-4 4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M6 16v2a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-2" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </div>
        {file ? (
          <div className="space-y-1">
            <p className="text-sm font-medium text-emerald-700">Uploaded</p>
            <p className="text-sm text-slate-700">{file.name}</p>
            <p className="text-xs text-slate-500">Click to replace</p>
          </div>
        ) : (
          <div className="space-y-1">
            <p className="text-sm text-slate-700">{isDragActive ? 'Drop the file here' : 'Drag & drop or click to select'}</p>
            <p className="text-xs text-slate-500">We do not store your files</p>
          </div>
        )}
      </div>
    </div>
  )

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 px-6 py-10">
        <header className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-slate-900/30 backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-blue-200/80">US Space Command</p>
              <h1 className="text-3xl font-semibold leading-tight text-white sm:text-4xl">Sustainment Prediction Dashboard</h1>
              <p className="text-sm text-slate-200/80">
                Upload project & inventory files to simulate Facility Condition Index (FCI) scenarios until failure.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-200">
                API: 8000
              </span>
              <span className="rounded-full border border-blue-400/40 bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-200">
                App: {appPort}
              </span>
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-200/70">Step 1</p>
              <p className="text-sm font-semibold text-white">Upload Projects</p>
              <p className="text-xs text-slate-300/70">CSV/XLSX, includes fiscal year & scope</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-200/70">Step 2</p>
              <p className="text-sm font-semibold text-white">Upload Key/Inventory</p>
              <p className="text-xs text-slate-300/70">Systems + life expectancy</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-200/70">Step 3</p>
              <p className="text-sm font-semibold text-white">Run Prediction</p>
              <p className="text-xs text-slate-300/70">Markov-based simulation to failure</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-slate-200/70">Step 4</p>
              <p className="text-sm font-semibold text-white">Compare Scenarios</p>
              <p className="text-xs text-slate-300/70">Funded vs Unfunded FCI</p>
            </div>
          </div>
        </header>

        {!predictionResult ? (
          <div className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              {renderUploadCard(
                'Projects File',
                'Projects & scope',
                projectsFile,
                getProjectsRootProps,
                getProjectsInputProps,
                isProjectsDragActive
              )}
              {renderUploadCard(
                'Inventory / Key File',
                'Systems & life expectancy',
                inventoryFile,
                getInventoryRootProps,
                getInventoryInputProps,
                isInventoryDragActive
              )}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/5 p-4 shadow-inner shadow-slate-900/20">
              <div className="space-y-1">
                <p className="text-sm text-slate-200/80">Upload both files, then run a funded vs unfunded simulation to failure.</p>
                <p className="text-xs text-slate-300/60">We parse columns dynamically – no hard-coded headers required.</p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handlePredict}
                  disabled={loading || !projectsFile || !inventoryFile}
                  className={`rounded-xl px-6 py-3 text-sm font-semibold shadow-lg transition focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-400 focus:ring-offset-slate-900 ${
                    loading || !projectsFile || !inventoryFile
                      ? 'cursor-not-allowed bg-slate-600/60 text-slate-300'
                      : 'bg-blue-500 text-white hover:bg-blue-600'
                  }`}
                >
                  {loading ? 'Processing...' : 'Run Prediction'}
                </button>
                <button
                  onClick={handleReset}
                  className="rounded-xl border border-white/20 px-4 py-3 text-sm font-semibold text-white hover:border-white/40 hover:bg-white/10"
                >
                  Reset
                </button>
              </div>
            </div>

            {error && (
              <div className="rounded-xl border border-red-400/30 bg-red-500/10 p-4 text-sm text-red-50">
                {error}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-2xl border border-emerald-400/30 bg-emerald-500/10 p-4 shadow-inner shadow-emerald-800/30">
                <p className="text-xs text-emerald-100/80">Funded start</p>
                <p className="text-2xl font-semibold text-white">{summary?.fundedStart}%</p>
              </div>
              <div className="rounded-2xl border border-emerald-400/30 bg-emerald-500/10 p-4 shadow-inner shadow-emerald-800/30">
                <p className="text-xs text-emerald-100/80">Funded end (Year 10)</p>
                <p className="text-2xl font-semibold text-white">{summary?.fundedEnd}%</p>
              </div>
              <div className="rounded-2xl border border-blue-400/30 bg-blue-500/10 p-4 shadow-inner shadow-blue-800/30">
                <p className="text-xs text-blue-100/80">Average funded FCI</p>
                <p className="text-2xl font-semibold text-white">{summary?.avgFunded}%</p>
              </div>
              <div className="rounded-2xl border border-amber-400/30 bg-amber-500/10 p-4 shadow-inner shadow-amber-800/30">
                <p className="text-xs text-amber-100/80">Funded lift vs unfunded</p>
                <p className="text-2xl font-semibold text-white">{summary?.lift}%</p>
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-slate-900/30 backdrop-blur">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <h2 className="text-xl font-semibold text-white">
                  {viewMode === 'facility' ? 'Facility Average FCI' : 'Individual Component FCI Predictions'}
                </h2>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setViewMode('individual')}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      viewMode === 'individual'
                        ? 'bg-blue-500 text-white'
                        : 'bg-white/10 text-slate-300 hover:bg-white/20'
                    }`}
                  >
                    Individual Components
                  </button>
                  <button
                    onClick={() => setViewMode('facility')}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      viewMode === 'facility'
                        ? 'bg-blue-500 text-white'
                        : 'bg-white/10 text-slate-300 hover:bg-white/20'
                    }`}
                  >
                    Facility Average
                  </button>
                </div>
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-300/70">
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                  Years simulated: {predictionResult.results.years_simulated}
                </span>
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                  Start year: {predictionResult.results.start_year}
                </span>
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                  {Object.keys(predictionResult.systems).length} components tracked
                </span>
              </div>
              
              {viewMode === 'individual' && (
                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    onClick={() => setSelectedSystem(null)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      selectedSystem === null
                        ? 'bg-emerald-500 text-white'
                        : 'bg-white/10 text-slate-300 hover:bg-white/20'
                    }`}
                  >
                    All Components
                  </button>
                  {Object.keys(predictionResult.systems).map((system) => (
                    <button
                      key={system}
                      onClick={() => setSelectedSystem(system)}
                      className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                        selectedSystem === system
                          ? 'bg-emerald-500 text-white'
                          : 'bg-white/10 text-slate-300 hover:bg-white/20'
                      }`}
                    >
                      {system}
                    </button>
                  ))}
                </div>
              )}

              <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                <ResponsiveContainer width="100%" height={420}>
                  {viewMode === 'facility' ? (
                    <LineChart data={facilityChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        dataKey="year"
                        tick={{ fill: '#cbd5e1' }}
                        label={{ value: 'Year', position: 'insideBottom', offset: -5, fill: '#cbd5e1' }}
                      />
                      <YAxis
                        tick={{ fill: '#cbd5e1' }}
                        label={{ value: 'Average FCI', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }}
                        domain={[0, 100]}
                      />
                      <Tooltip
                        contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', color: '#e2e8f0' }}
                        labelStyle={{ color: '#e2e8f0' }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="Unfunded Scenario" stroke="#f97316" strokeWidth={2.4} dot={{ r: 3 }} />
                      <Line type="monotone" dataKey="Funded Scenario" stroke="#22d3ee" strokeWidth={2.4} dot={{ r: 3 }} />
                    </LineChart>
                  ) : selectedSystem ? (
                    <LineChart data={selectedSystemChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        dataKey="year"
                        tick={{ fill: '#cbd5e1' }}
                        label={{ value: 'Year', position: 'insideBottom', offset: -5, fill: '#cbd5e1' }}
                      />
                      <YAxis
                        tick={{ fill: '#cbd5e1' }}
                        label={{ value: `${selectedSystem} FCI`, angle: -90, position: 'insideLeft', fill: '#cbd5e1' }}
                        domain={[0, 100]}
                      />
                      <Tooltip
                        contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', color: '#e2e8f0' }}
                        labelStyle={{ color: '#e2e8f0' }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="Unfunded" stroke="#f97316" strokeWidth={2.4} dot={{ r: 3 }} />
                      <Line type="monotone" dataKey="Funded" stroke="#22d3ee" strokeWidth={2.4} dot={{ r: 3 }} />
                    </LineChart>
                  ) : (
                    <LineChart data={allSystemsChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        dataKey="year"
                        tick={{ fill: '#cbd5e1' }}
                        label={{ value: 'Year', position: 'insideBottom', offset: -5, fill: '#cbd5e1' }}
                      />
                      <YAxis
                        tick={{ fill: '#cbd5e1' }}
                        label={{ value: 'Component FCI', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }}
                        domain={[0, 100]}
                      />
                      <Tooltip
                        contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', color: '#e2e8f0' }}
                        labelStyle={{ color: '#e2e8f0' }}
                      />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      {Object.keys(predictionResult.systems).map((system, index) => (
                        <Line
                          key={system}
                          type="monotone"
                          dataKey={system}
                          stroke={SYSTEM_COLORS[index % SYSTEM_COLORS.length]}
                          strokeWidth={1.8}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  )}
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-lg shadow-slate-900/25 backdrop-blur">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Parsing Summary</h3>
                  <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-100">Data checks</span>
                </div>
                <div className="grid gap-3 text-sm text-slate-200">
                  <div className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-950/40 px-4 py-3">
                    <span>Projects rows parsed</span>
                    <span className="font-medium text-blue-200">{predictionResult.stats.projects_rows}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-950/40 px-4 py-3">
                    <span>Inventory rows parsed</span>
                    <span className="font-medium text-blue-200">{predictionResult.stats.inventory_rows}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-950/40 px-4 py-3">
                    <span>Project years detected</span>
                    <span className="font-medium text-blue-200">{predictionResult.stats.project_years}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-950/40 px-4 py-3">
                    <span>Systems detected</span>
                    <span className="font-medium text-blue-200">{predictionResult.stats.systems_count}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-950/40 px-4 py-3">
                    <span>Average life expectancy</span>
                    <span className="font-medium text-blue-200">{predictionResult.stats.avg_life_expectancy} yrs</span>
                  </div>
                </div>
                <p className="mt-3 text-xs text-slate-300/70">
                  Use these checks to confirm the dashboard is reading the right columns and year counts.
                </p>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-lg shadow-slate-900/25 backdrop-blur">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold text-white">Markov Transitions</h3>
                    <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-100">Real Data</span>
                  </div>
                  <div className="flex gap-2 text-xs">
                    <button
                      onClick={() => setShowMarkovMode('diagram')}
                      className={`rounded-lg px-3 py-1.5 font-semibold transition ${
                        showMarkovMode === 'diagram' ? 'bg-blue-500 text-white' : 'bg-white/10 text-slate-200 hover:bg-white/20'
                      }`}
                    >
                      Diagram
                    </button>
                    <button
                      onClick={() => setShowMarkovMode('table')}
                      className={`rounded-lg px-3 py-1.5 font-semibold transition ${
                        showMarkovMode === 'table' ? 'bg-blue-500 text-white' : 'bg-white/10 text-slate-200 hover:bg-white/20'
                      }`}
                    >
                      Table
                    </button>
                  </div>
                </div>
                {(megaPlantUmlSvg || megaPlantUmlUrl) && (
                  <div className="mb-4 rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                    <div className="mb-2 flex items-center justify-between">
                      <h4 className="text-sm font-semibold text-blue-200">Comprehensive Diagram (all systems & projects)</h4>
                      <span className="text-xs text-slate-400">Rendered via PlantUML</span>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-slate-900/60 p-3 max-h-[520px] overflow-auto">
                      {megaPlantUmlSvg ? (
                        <div
                          className="w-full"
                          dangerouslySetInnerHTML={{ __html: megaPlantUmlSvg }}
                        />
                      ) : (
                        <img
                          src={megaPlantUmlUrl!}
                          alt="Comprehensive Markov diagram"
                          className="w-full object-contain"
                          loading="lazy"
                        />
                      )}
                    </div>
                  </div>
                )}
                <div className="space-y-4">
                  {aggregateMarkovMatrix && (
                    <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                      <div className="mb-3 flex items-center justify-between">
                        <h4 className="text-sm font-semibold text-blue-200">All Components (combined)</h4>
                        <div className="flex gap-2 text-xs text-slate-400">
                          <span className="rounded bg-white/5 px-2 py-0.5">
                            {aggregateMarkovMatrix.degradation_rate_per_year}%/yr
                          </span>
                          <span className="rounded bg-white/5 px-2 py-0.5">
                            {aggregateMarkovMatrix.years_to_failure} yr avg life
                          </span>
                        </div>
                      </div>
                      {showMarkovMode === 'diagram' ? (() => {
                        const diagramUrl = buildPlantUmlUrl('All Components', aggregateMarkovMatrix, { title: 'All Components (Average)' })
                        return (
                          <div className="rounded-xl border border-white/10 bg-slate-900/60 p-3">
                            {diagramUrl ? (
                              <img
                                src={diagramUrl}
                                alt="All components Markov diagram"
                                className="w-full max-h-[360px] object-contain bg-slate-950/40 rounded-lg"
                              />
                            ) : (
                              <div className="flex h-[360px] items-center justify-center text-slate-400">
                                <div className="text-center">
                                  <div className="mb-2 text-sm">Loading diagram...</div>
                                  <div className="text-xs">Rendering via PlantUML</div>
                                </div>
                              </div>
                            )}
                            <p className="mt-2 text-[11px] text-slate-400">
                              Combined diagram using average transition probabilities across all components.
                            </p>
                          </div>
                        )
                      })() : (
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="text-slate-400">
                                <th className="px-2 py-1 text-left">From \\ To</th>
                                {aggregateMarkovMatrix.states.map((state) => (
                                  <th key={state} className="px-2 py-1 text-center">{state}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {aggregateMarkovMatrix.states.map((fromState, rowIndex) => (
                                <tr key={fromState} className="border-t border-white/5">
                                  <td className="px-2 py-1.5 font-medium text-slate-300">{fromState}</td>
                                  {aggregateMarkovMatrix.matrix[rowIndex].map((prob, colIndex) => (
                                    <td
                                      key={colIndex}
                                      className={`px-2 py-1.5 text-center ${
                                        prob > 0.5 ? 'text-emerald-300 font-medium' :
                                        prob > 0.1 ? 'text-amber-300' :
                                        prob > 0 ? 'text-slate-400' : 'text-slate-600'
                                      }`}
                                    >
                                      {prob.toFixed(3)}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="max-h-[520px] overflow-y-auto space-y-4 pr-1">
                    {Object.entries(predictionResult.markov_matrices).map(([system, matrix]) => {
                      const diagramUrl = buildPlantUmlUrl(system, matrix)
                      const desc = getSystemDescription(system)
                      return (
                        <div key={system} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                          <div className="mb-3 flex items-center justify-between">
                            <div>
                              <h4 className="text-sm font-semibold text-emerald-200">{system}</h4>
                              <p className="mt-1 text-[10px] text-slate-400 max-w-md">
                                {desc.impact}
                              </p>
                            </div>
                            <div className="flex gap-2 text-xs text-slate-400">
                              <span className="rounded bg-white/5 px-2 py-0.5">
                                {matrix.degradation_rate_per_year}%/yr
                              </span>
                              <span className="rounded bg-white/5 px-2 py-0.5">
                                {matrix.years_to_failure} yr life
                              </span>
                            </div>
                          </div>

                          {showMarkovMode === 'diagram' ? (
                            <div className="rounded-xl border border-white/10 bg-slate-900/60 p-3">
                              {diagramUrl ? (
                                <img
                                  src={diagramUrl}
                                  alt={`${system} Markov diagram`}
                                  className="w-full max-h-[320px] object-contain bg-slate-950/40 rounded-lg"
                                  loading="lazy"
                                />
                              ) : (
                                <div className="flex h-[320px] items-center justify-center text-slate-400">
                                  <div className="text-center">
                                    <div className="mb-2 text-sm">Loading diagram...</div>
                                    <div className="text-xs">Rendering via PlantUML</div>
                                  </div>
                                </div>
                              )}
                              <p className="mt-2 text-[11px] text-slate-400">
                                Rendered via PlantUML with per-year transition probabilities and state descriptions.
                              </p>
                            </div>
                          ) : (
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="text-slate-400">
                                    <th className="px-2 py-1 text-left">From \\ To</th>
                                    {matrix.states.map((state) => (
                                      <th key={state} className="px-2 py-1 text-center">{state}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {matrix.states.map((fromState, rowIndex) => (
                                    <tr key={fromState} className="border-t border-white/5">
                                      <td className="px-2 py-1.5 font-medium text-slate-300">{fromState}</td>
                                      {matrix.matrix[rowIndex].map((prob, colIndex) => (
                                        <td
                                          key={colIndex}
                                          className={`px-2 py-1.5 text-center ${
                                            prob > 0.5 ? 'text-emerald-300 font-medium' :
                                            prob > 0.1 ? 'text-amber-300' :
                                            prob > 0 ? 'text-slate-400' : 'text-slate-600'
                                          }`}
                                        >
                                          {prob.toFixed(3)}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
                <p className="mt-3 text-xs text-slate-300/70">
                  Real transition probabilities calculated from each component&apos;s life expectancy. Toggle diagram/table.
                </p>
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-lg shadow-slate-900/25 backdrop-blur">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Component Status Summary</h3>
                  <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-100">
                    {Object.keys(predictionResult.systems).length} components
                  </span>
                </div>
                <div className="divide-y divide-white/5 rounded-2xl border border-white/10 bg-slate-950/40 max-h-80 overflow-y-auto">
                  {Object.entries(predictionResult.systems).map(([system, lifeExp]) => {
                    const unfundedHistory = predictionResult.results.system_predictions_unfunded?.[system]
                    const fundedHistory = predictionResult.results.system_predictions_funded?.[system]
                    const endUnfunded = unfundedHistory?.[unfundedHistory.length - 1]
                    const endFunded = fundedHistory?.[fundedHistory.length - 1]
                    
                    const getStateColor = (state: string) => {
                      if (state === 'Good') return 'text-emerald-300 bg-emerald-500/20 border-emerald-400/40'
                      if (state === 'Fair') return 'text-amber-300 bg-amber-500/20 border-amber-400/40'
                      return 'text-red-300 bg-red-500/20 border-red-400/40'
                    }
                    
                    return (
                      <div key={system} className="px-4 py-3">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-slate-100">{system}</span>
                          <span className="text-xs text-slate-400">{lifeExp} yr life</span>
                        </div>
                        <div className="flex items-center gap-3 text-xs">
                          <div className="flex items-center gap-1">
                            <span className="text-slate-500">Unfunded:</span>
                            <span className={`rounded-full border px-2 py-0.5 ${getStateColor(endUnfunded?.state || 'Poor')}`}>
                              {endUnfunded?.fci?.toFixed(1)}% ({endUnfunded?.state})
                            </span>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="text-slate-500">Funded:</span>
                            <span className={`rounded-full border px-2 py-0.5 ${getStateColor(endFunded?.state || 'Poor')}`}>
                              {endFunded?.fci?.toFixed(1)}% ({endFunded?.state})
                            </span>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
                <p className="mt-2 text-xs text-slate-300/70">
                  Shows each component&apos;s predicted condition at year {predictionResult.results.start_year + predictionResult.results.years_simulated - 1}
                </p>
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-lg shadow-slate-900/25 backdrop-blur">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Projects by Year</h3>
                  <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-100">
                    {Object.keys(predictionResult.projects_by_year).length || 'No'} years
                  </span>
                </div>
                <div className="grid max-h-72 grid-cols-1 gap-2 overflow-y-auto pr-1">
                  {Object.entries(predictionResult.projects_by_year).map(([year, projects]) => (
                    <div
                      key={year}
                      className="flex items-center justify-between rounded-xl border border-white/10 bg-slate-950/40 px-4 py-3 text-sm text-slate-100"
                    >
                      <span className="font-medium">{year}</span>
                      <span className="text-slate-300">{projects.length} project(s)</span>
                    </div>
                  ))}
                  {Object.keys(predictionResult.projects_by_year).length === 0 && (
                    <p className="text-slate-200 text-sm">No projects found in the uploaded file.</p>
                  )}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap justify-between gap-3">
              <button
                onClick={handleReset}
                className="rounded-xl border border-white/20 px-6 py-3 text-sm font-semibold text-white hover:border-white/40 hover:bg-white/10"
              >
                Upload New Files
              </button>
              <div className="text-xs text-slate-300/70">
                Tip: If you see 404s for assets, open the app on the port shown above (often 3001 when 3000 is in use).
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
