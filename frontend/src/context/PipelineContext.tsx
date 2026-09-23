import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import type { PipelineType, ProjectDetail, RecommenderHandoffResponse } from '../api/types'

interface PipelineState {
  fileName: string
  fileType: string
  fileContent: string
  model: string
  pipeline: PipelineType
  userPrompt: string
  changeApplied: boolean
  humanIntervention: boolean
  projectId: number | null
  recommenderJobId: string | null
  trimmerJobId: string | null
  siloJobId: string | null
  muloJobId: string | null
  handoff: RecommenderHandoffResponse | null
  trimmingParams: Record<string, unknown>
  statesInputs: string[]
}

interface PipelineContextValue extends PipelineState {
  setFile: (name: string, type: string, content: string) => void
  setModel: (model: string) => void
  setPipeline: (pipeline: PipelineType) => void
  setUserPrompt: (userPrompt: string) => void
  setProjectId: (projectId: number | null) => void
  setRegularizeResult: (content: string, changeApplied: boolean, humanIntervention: boolean) => void
  setRecommenderJobId: (jobId: string | null) => void
  setTrimmerJobId: (jobId: string | null) => void
  setSiloJobId: (jobId: string | null) => void
  setMuloJobId: (jobId: string | null) => void
  setHandoff: (handoff: RecommenderHandoffResponse | null) => void
  setTrimmingParams: (params: Record<string, unknown>) => void
  setStatesInputs: (inputs: string[]) => void
  updateFileContent: (content: string) => void
  clearDesignJobs: () => void
  hydrateFromProject: (project: ProjectDetail) => void
  reset: () => void
}

const initialState: PipelineState = {
  fileName: '',
  fileType: 'python',
  fileContent: '',
  model: 'gpt-4o',
  pipeline: null,
  userPrompt: '',
  changeApplied: false,
  humanIntervention: false,
  projectId: null,
  recommenderJobId: null,
  trimmerJobId: null,
  siloJobId: null,
  muloJobId: null,
  handoff: null,
  trimmingParams: {},
  statesInputs: [],
}

const PipelineContext = createContext<PipelineContextValue | null>(null)

export function PipelineProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PipelineState>(initialState)

  const value = useMemo<PipelineContextValue>(
    () => ({
      ...state,
      setFile: (fileName, fileType, fileContent) =>
        setState((prev) => ({ ...prev, fileName, fileType, fileContent, projectId: null })),
      setModel: (model) => setState((prev) => ({ ...prev, model })),
      setPipeline: (pipeline) => setState((prev) => ({ ...prev, pipeline })),
      setUserPrompt: (userPrompt) => setState((prev) => ({ ...prev, userPrompt })),
      setProjectId: (projectId) => setState((prev) => ({ ...prev, projectId })),
      setRegularizeResult: (fileContent, changeApplied, humanIntervention) =>
        setState((prev) => ({
          ...prev,
          fileContent,
          // Regularizer always outputs Python (MATLAB is converted).
          fileType: 'python',
          changeApplied,
          humanIntervention,
        })),
      setRecommenderJobId: (recommenderJobId) =>
        setState((prev) => ({ ...prev, recommenderJobId })),
      setTrimmerJobId: (trimmerJobId) => setState((prev) => ({ ...prev, trimmerJobId })),
      setSiloJobId: (siloJobId) => setState((prev) => ({ ...prev, siloJobId })),
      setMuloJobId: (muloJobId) => setState((prev) => ({ ...prev, muloJobId })),
      setHandoff: (handoff) =>
        setState((prev) => ({
          ...prev,
          handoff,
          fileContent: handoff?.file_content ?? prev.fileContent,
          statesInputs: handoff?.states_inputs ?? prev.statesInputs,
        })),
      setTrimmingParams: (trimmingParams) =>
        setState((prev) => ({ ...prev, trimmingParams })),
      setStatesInputs: (statesInputs) => setState((prev) => ({ ...prev, statesInputs })),
      updateFileContent: (fileContent) => setState((prev) => ({ ...prev, fileContent })),
      clearDesignJobs: () =>
        setState((prev) => ({
          ...prev,
          recommenderJobId: null,
          trimmerJobId: null,
          siloJobId: null,
          muloJobId: null,
          handoff: null,
          trimmingParams: {},
          statesInputs: [],
        })),
      hydrateFromProject: (project) =>
        setState({
          ...initialState,
          fileName: project.file_name || project.title || '',
          fileType: project.file_type || 'python',
          fileContent: project.file_content || '',
          model: project.llm_model || initialState.model,
          pipeline: project.pipeline_type,
          projectId: project.id,
        }),
      reset: () => setState(initialState),
    }),
    [state],
  )

  return <PipelineContext.Provider value={value}>{children}</PipelineContext.Provider>
}

export function usePipeline() {
  const context = useContext(PipelineContext)
  if (!context) {
    throw new Error('usePipeline must be used within PipelineProvider')
  }
  return context
}
