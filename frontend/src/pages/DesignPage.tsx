import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { PlantModelResult, PreLaunchConfig } from '../api/types'
import { AgentPlantChat } from '../components/AgentPlantChat'
import { PreLaunchModal } from '../components/PreLaunchModal'
import { usePipeline } from '../context/PipelineContext'

export function DesignPage() {
  const navigate = useNavigate()
  const pipeline = usePipeline()
  const [activeResult, setActiveResult] = useState<PlantModelResult | null>(null)
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [isPreLaunchOpen, setIsPreLaunchOpen] = useState(false)

  const handleUseModel = (result: PlantModelResult, convId?: number | null) => {
    setActiveResult(result)
    setConversationId(convId ?? null)
    setIsPreLaunchOpen(true)
  }

  const handlePreLaunchSuccess = (artifactId: string, preLaunch: PreLaunchConfig) => {
    if (!activeResult) return
    const safeName = activeResult.system_name.trim().replace(/[^\w\-]+/g, '_') || 'dynamics'
    pipeline.setFile(`${safeName}.py`, 'python', activeResult.python_code)
    sessionStorage.setItem('labcd_last_artifact_id', artifactId)
    sessionStorage.setItem('labcd_last_pre_launch', JSON.stringify(preLaunch))
    const convParam = conversationId ? `&new=${conversationId}` : ''
    navigate(`/case-studies?artifact_id=${encodeURIComponent(artifactId)}${convParam}`)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col h-full">
      <AgentPlantChat model="gpt-4o-mini" onUseModel={handleUseModel} />

      {activeResult && (
        <PreLaunchModal
          isOpen={isPreLaunchOpen}
          onClose={() => setIsPreLaunchOpen(false)}
          plant={{
            system_name: activeResult.system_name,
            python_code: activeResult.python_code,
            metadata: activeResult.metadata as Record<string, unknown>,
          }}
          conversationId={conversationId}
          systemName={activeResult.system_name}
          onSuccess={handlePreLaunchSuccess}
        />
      )}
    </div>
  )
}
