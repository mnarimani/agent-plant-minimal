/**
 * Minimal scaffolding App for isolated AgentPlant / PlantModelChat UI.
 */
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { PipelineProvider } from './context/PipelineContext'
import { DesignPage } from './pages/DesignPage'
import './index.css'

export default function App() {
  return (
    <AuthProvider>
      <PipelineProvider>
        <BrowserRouter>
          <div className="flex h-full min-h-0 flex-1 flex-col" style={{ height: '100%', minHeight: '100dvh' }}>
            <Routes>
              <Route path="/" element={<DesignPage />} />
              <Route path="/design" element={<DesignPage />} />
              <Route path="*" element={<DesignPage />} />
            </Routes>
          </div>
        </BrowserRouter>
      </PipelineProvider>
    </AuthProvider>
  )
}
