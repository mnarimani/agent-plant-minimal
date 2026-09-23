import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { authApi } from '../api/endpoints'
import { ApiError, clearAuthToken, getAuthToken, setAuthToken } from '../api/client'
import type { AuthUser, MessageResponse } from '../api/types'

interface AuthContextValue {
  user: AuthUser | null
  token: string | null
  loading: boolean
  sessionError: boolean
  login: (email: string, password: string) => Promise<void>
  loginWithToken: (accessToken: string) => Promise<void>
  register: (
    email: string,
    password: string,
    referralCode?: string | null,
  ) => Promise<MessageResponse>
  logout: () => Promise<void>
  hasAction: (code: string) => boolean
  canUsePipeline: (pipeline: 'siloDesign' | 'muloDesign') => boolean
  refreshUser: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

const PIPELINE_ACTION: Record<'siloDesign' | 'muloDesign', string> = {
  siloDesign: 'pipeline:silo',
  muloDesign: 'pipeline:mulo',
}

function isUnauthorized(err: unknown): boolean {
  return err instanceof ApiError && err.status === 401
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => window.setTimeout(resolve, ms))
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [token, setToken] = useState<string | null>(() => getAuthToken())
  const [loading, setLoading] = useState(true)
  const [sessionError, setSessionError] = useState(false)

  const refreshUser = useCallback(async () => {
    const current = getAuthToken()
    if (!current) {
      setUser(null)
      setToken(null)
      setSessionError(false)
      setLoading(false)
      return
    }

    setLoading(true)
    setSessionError(false)
    const maxAttempts = 2

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const me = await authApi.me()
        setUser(me)
        setToken(current)
        setSessionError(false)
        setLoading(false)
        return
      } catch (err) {
        if (isUnauthorized(err)) {
          clearAuthToken()
          setUser(null)
          setToken(null)
          setSessionError(false)
          setLoading(false)
          return
        }
        if (attempt < maxAttempts) {
          await sleep(500)
          continue
        }
        // Keep token on transient failures so the user can retry without re-entering password.
        setSessionError(true)
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    void refreshUser()
  }, [refreshUser])

  const applyToken = useCallback(async (accessToken: string) => {
    setAuthToken(accessToken)
    setToken(accessToken)
    try {
      const me = await authApi.me()
      setUser(me)
      setSessionError(false)
    } catch (err) {
      clearAuthToken()
      setToken(null)
      setUser(null)
      throw err
    }
  }, [])

  const login = useCallback(
    async (email: string, password: string) => {
      const result = await authApi.login({ email, password })
      await applyToken(result.access_token)
    },
    [applyToken],
  )

  const loginWithToken = useCallback(
    async (accessToken: string) => {
      await applyToken(accessToken)
    },
    [applyToken],
  )

  const register = useCallback(
    async (email: string, password: string, referralCode?: string | null) => {
      return authApi.register({
        email,
        password,
        ...(referralCode ? { referral_code: referralCode } : {}),
      })
    },
    [],
  )

  const logout = useCallback(async () => {
    try {
      if (getAuthToken()) {
        await authApi.logout()
      }
    } catch {
      // Still clear local session if the server revoke fails.
    }
    clearAuthToken()
    setToken(null)
    setUser(null)
    setSessionError(false)
  }, [])

  const hasAction = useCallback(
    (code: string) => {
      if (!user) return false
      return user.actions.includes(code)
    },
    [user],
  )

  const canUsePipeline = useCallback(
    (pipeline: 'siloDesign' | 'muloDesign') => hasAction(PIPELINE_ACTION[pipeline]),
    [hasAction],
  )

  const value = useMemo(
    () => ({
      user,
      token,
      loading,
      sessionError,
      login,
      loginWithToken,
      register,
      logout,
      hasAction,
      canUsePipeline,
      refreshUser,
    }),
    [
      user,
      token,
      loading,
      sessionError,
      login,
      loginWithToken,
      register,
      logout,
      hasAction,
      canUsePipeline,
      refreshUser,
    ],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return ctx
}
