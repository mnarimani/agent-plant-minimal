import { useEffect, useState } from 'react'
import { ArrowRight } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { healthApi } from '../api/endpoints'
import type { PlantModelResult, PreLaunchConfig } from '../api/types'
import { PlantModelChat } from '../components/PlantModelChat'
import { PreLaunchModal } from '../components/PreLaunchModal'
import { usePipeline } from '../context/PipelineContext'
import { AUTO_MODEL } from '../lib/modelPicker'

export function DesignPage() {
  const navigate = useNavigate()
  const pipeline = usePipeline()
  const [models, setModels] = useState<string[]>(['gpt-4o', 'gpt-4o-mini'])
  const [activeResult, setActiveResult] = useState<PlantModelResult | null>(null)
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [isPreLaunchOpen, setIsPreLaunchOpen] = useState(false)

  useEffect(() => {
    healthApi.models().then((res) => setModels(res.llm_models)).catch(() => {})
  }, [])

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
    <div className="flex min-h-0 flex-1 flex-col">
      <PlantModelChat
        model={AUTO_MODEL}
        models={models}
        onModelChange={pipeline.setModel}
        onUseModel={handleUseModel}
        continueLabel="Configure & launch →"
        continueIcon={<ArrowRight className="size-3.5" aria-hidden />}
        isModalOpen={isPreLaunchOpen}
      />

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

