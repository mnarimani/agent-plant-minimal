/**
 * Minimal scaffolding App for isolated AgentPlant / PlantModelChat UI.
 * Original LabCD_Application App.tsx is not modified; this is a new thin shell
 * that only mounts the DesignPage (AgentPlant chat) for standalone R&D.
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
          <Routes>
            <Route path="/" element={<DesignPage />} />
            <Route path="/design" element={<DesignPage />} />
            <Route path="*" element={<DesignPage />} />
          </Routes>
        </BrowserRouter>
      </PipelineProvider>
    </AuthProvider>
  )
}
