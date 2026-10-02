import { apiFetch, artifactUrl, AUTH_TIMEOUT_MS, getAuthToken, projectArtifactUrl, streamEvents } from './client'
import type {
  ActionInfo,
  ArtifactResponse,
  AdminUserCredits,
  AdminUserDetail,
  ApiKeysResponse,
  ApiKeysUpdate,
  AuthUser,
  BillingStatus,
  CaseStudiesResponse,
  ControlDesignTemplate,
  CreditDashboard,
  CreditLedgerEntry,
  CreditSettings,
  CreditSettingsUpdate,
  CreditUsageSession,
  DefaultPlanInfo,
  ErrorEvent,
  ErrorTrackingSettings,
  AuditLogEntry,
  BeforeTestSurveyPayload,
  BeforeTestSurveyResponseRow,
  FeedbackSurveyRequest,
  FeedbackSurveyResponseRow,
  JobResponse,
  JobStatusResponse,
  ModelsResponse,
  MonitoringResponse,
  AnalyticsResponse,
  TelegramAnalyticsSettings,
  TelegramAnalyticsSettingsUpdate,
  TelegramAnalyticsTestResult,
  MuloDesignerStateResponse,
  MuloSimulateResponse,
  OutroSurveyPayload,
  PlanInfo,
  PlantModelChatMessage,
  PlantModelChatResponse,
  PlantModelConversationDetail,
  PlantModelConversationSummary,
  PlantModelSessionState,
  ProfileSurveyRequest,
  ProjectDetail,
  ProjectPipelineType,
  ProjectSummary,
  PublicPlan,
  RagStatusResponse,
  RecommenderHandoffResponse,
  RegularizeResponse,
  RoleInfo,
  SiloSimulateResponse,
  StandardizeResponse,
  SurveyResponses,
  SurveySettings,
  SurveyStatus,
  TokenResponse,
  TrimmerArtifactsResponse,
  TutorialDocument,
  TutorialDocumentSummary,
  TutorialVideo,
  UploadResponse,
  UserJourney,
  MediaUploadResponse,
  SiteBrand,
  NavMenuItem,
  LandingPayload,
  BlogPost,
  BlogPostListItem,
  BugReport,
  BugReportSettings,
  AuthSessionInfo,
  MessageResponse,
  SsoProviderAdmin,
  SsoProviderCreate,
  SsoProviderPublic,
  SsoProviderUpdate,
  AdaptiveClarifyRequest,
  AdaptiveClarifyResponse,
  AdaptiveJobCreateRequest,
  AdaptiveJobCreateResponse,
  AdaptiveJobResultsResponse,
  AdaptiveJobStatusResponse,
  AdaptiveJobSummary,
  AdaptiveDiagnosisChatRequest,
  AdaptiveDiagnosisChatResponse,
  MPCDiagnosticsRequest,
  MPCDiagnosticsResponse,
  MPCSimulateRequest,
  MPCSimulateResponse,
  MPCJobCreateRequest,
  MPCJobCreateResponse,
  MPCJobResultsResponse,
  MPCJobStatusResponse,
  MPCJobSummary,
  ArtifactCreateRequest,
  ArtifactCreateResponse,
  ArtifactDetail,
  ArtifactPluginResponse,
  ArtifactSummary,
  ValidationRequest,
  ValidationResponse,
} from './types'
import { viewerTimeZone } from '../lib/formatDateTime'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

function buildQuery(params?: Record<string, string | number | boolean | undefined | null>): string {
  const query = new URLSearchParams()
  if (!params) return ''
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    query.set(key, String(value))
  }
  const suffix = query.toString()
  return suffix ? `?${suffix}` : ''
}

export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export async function downloadAdminCsv(
  path: string,
  params?: Record<string, string | number | boolean | undefined | null>,
): Promise<Blob> {
  const token = getAuthToken()
  const suffix = buildQuery(params)
  const response = await fetch(`${API_BASE}${path}${suffix}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!response.ok) {
    throw new Error('Failed to download CSV')
  }
  return response.blob()
}

export const authApi = {
  login: (body: { email: string; password: string }) =>
    apiFetch<TokenResponse>(
      '/auth/login',
      {
        method: 'POST',
        body: JSON.stringify(body),
      },
      AUTH_TIMEOUT_MS,
    ),
  register: (body: { email: string; password: string; referral_code?: string | null }) =>
    apiFetch<MessageResponse>(
      '/auth/register',
      {
        method: 'POST',
        body: JSON.stringify(body),
      },
      AUTH_TIMEOUT_MS,
    ),
  verifyEmail: (body: { token: string }) =>
    apiFetch<MessageResponse>(
      '/auth/verify-email',
      {
        method: 'POST',
        body: JSON.stringify(body),
      },
      AUTH_TIMEOUT_MS,
    ),
  resendVerification: (body: { email: string }) =>
    apiFetch<MessageResponse>(
      '/auth/resend-verification',
      {
        method: 'POST',
        body: JSON.stringify(body),
      },
      AUTH_TIMEOUT_MS,
    ),
  forgotPassword: (body: { email: string }) =>
    apiFetch<MessageResponse>(
      '/auth/forgot-password',
      {
        method: 'POST',
        body: JSON.stringify(body),
      },
      AUTH_TIMEOUT_MS,
    ),
  resetPassword: (body: { token: string; new_password: string }) =>
    apiFetch<MessageResponse>(
      '/auth/reset-password',
      {
        method: 'POST',
        body: JSON.stringify(body),
      },
      AUTH_TIMEOUT_MS,
    ),
  logout: () =>
    apiFetch<void>('/auth/logout', { method: 'POST' }, AUTH_TIMEOUT_MS),
  listSessions: () => apiFetch<AuthSessionInfo[]>('/auth/sessions'),
  revokeSession: (sessionId: number) =>
    apiFetch<void>(`/auth/sessions/${sessionId}`, { method: 'DELETE' }),
  me: () => apiFetch<AuthUser>('/auth/me', {}, AUTH_TIMEOUT_MS),
  updateProfile: (body: {
    display_name?: string | null
    email?: string
    theme?: AuthUser['theme']
    current_password?: string
  }) =>
    apiFetch<AuthUser>('/auth/me', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  changePassword: (body: { current_password: string; new_password: string }) =>
    apiFetch<void>('/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  uploadAvatar: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return apiFetch<AuthUser>('/auth/me/avatar', { method: 'POST', body: form })
  },
  removeAvatar: () => apiFetch<AuthUser>('/auth/me/avatar', { method: 'DELETE' }),
  listSsoProviders: () =>
    apiFetch<SsoProviderPublic[]>('/auth/sso/providers', {}, AUTH_TIMEOUT_MS),
  ssoStartUrl: (provider: string, redirectTo?: string) => {
    const query = redirectTo
      ? `?redirect_to=${encodeURIComponent(redirectTo)}`
      : ''
    return `${API_BASE}/auth/sso/${encodeURIComponent(provider)}/start${query}`
  },
}

export const adminApi = {
  getMonitoring: () => apiFetch<MonitoringResponse>('/admin/monitoring'),
  getAnalytics: (days = 30, tz = viewerTimeZone()) => {
    const query = new URLSearchParams({
      days: String(days),
      tz,
    })
    return apiFetch<AnalyticsResponse>(`/admin/analytics?${query.toString()}`)
  },
  getTelegramAnalyticsSettings: () =>
    apiFetch<TelegramAnalyticsSettings>('/admin/analytics/telegram'),
  updateTelegramAnalyticsSettings: (body: TelegramAnalyticsSettingsUpdate) =>
    apiFetch<TelegramAnalyticsSettings>('/admin/analytics/telegram', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  testTelegramAnalyticsReport: () =>
    apiFetch<TelegramAnalyticsTestResult>('/admin/analytics/telegram/test', {
      method: 'POST',
    }),
  listActions: () => apiFetch<ActionInfo[]>('/admin/actions'),
  listPlans: (params?: { active_only?: boolean }) => {
    const query = new URLSearchParams()
    if (params?.active_only) query.set('active_only', 'true')
    const suffix = query.toString() ? `?${query}` : ''
    return apiFetch<PlanInfo[]>(`/admin/plans${suffix}`)
  },
  createPlan: (body: {
    name: string
    description?: string
    price?: number
    actions?: string[]
    models?: string[]
    is_active?: boolean
    plan_code?: string | null
    price_yearly?: number | null
    is_contact_sales?: boolean
    is_most_popular?: boolean
    stripe_product_id?: string | null
    stripe_price_id_monthly?: string | null
    stripe_price_id_yearly?: string | null
  }) =>
    apiFetch<PlanInfo>('/admin/plans', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updatePlan: (
    planId: number,
    body: {
      name?: string
      description?: string
      price?: number
      actions?: string[]
      models?: string[]
      is_active?: boolean
      plan_code?: string | null
      price_yearly?: number | null
      is_contact_sales?: boolean
      is_most_popular?: boolean
      stripe_product_id?: string | null
      stripe_price_id_monthly?: string | null
      stripe_price_id_yearly?: string | null
    },
  ) =>
    apiFetch<PlanInfo>(`/admin/plans/${planId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deletePlan: (planId: number) =>
    apiFetch<void>(`/admin/plans/${planId}`, { method: 'DELETE' }),
  listRoles: (params?: { active_only?: boolean }) => {
    const query = new URLSearchParams()
    if (params?.active_only) query.set('active_only', 'true')
    const suffix = query.toString() ? `?${query}` : ''
    return apiFetch<RoleInfo[]>(`/admin/roles${suffix}`)
  },
  createRole: (body: {
    name: string
    description?: string
    actions?: string[]
    is_active?: boolean
  }) =>
    apiFetch<RoleInfo>('/admin/roles', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateRole: (
    roleId: number,
    body: {
      name?: string
      description?: string
      actions?: string[]
      is_active?: boolean
    },
  ) =>
    apiFetch<RoleInfo>(`/admin/roles/${roleId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteRole: (roleId: number) =>
    apiFetch<void>(`/admin/roles/${roleId}`, { method: 'DELETE' }),
  getDefaultPlan: () => apiFetch<DefaultPlanInfo>('/admin/settings/default-plan'),
  setDefaultPlan: (planId: number) =>
    apiFetch<DefaultPlanInfo>('/admin/settings/default-plan', {
      method: 'PUT',
      body: JSON.stringify({ plan_id: planId }),
    }),
  listUsers: () => apiFetch<AuthUser[]>('/admin/users'),
  getUser: (userId: number) => apiFetch<AdminUserDetail>(`/admin/users/${userId}`),
  getUserJourney: (userId: number) =>
    apiFetch<UserJourney>(`/admin/users/${userId}/journey`),
  getUserCredits: (userId: number) =>
    apiFetch<AdminUserCredits>(`/admin/users/${userId}/credits`),
  adjustUserCredits: (userId: number, body: { amount: number; note?: string }) =>
    apiFetch<AdminUserCredits>(`/admin/users/${userId}/credits/adjust`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getCreditSettings: () => apiFetch<CreditSettings>('/admin/credits/settings'),
  updateCreditSettings: (body: CreditSettingsUpdate) =>
    apiFetch<CreditSettings>('/admin/credits/settings', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  revokeUserSession: (userId: number, sessionId: number) =>
    apiFetch<void>(`/admin/users/${userId}/sessions/${sessionId}`, { method: 'DELETE' }),
  createUser: (body: {
    email: string
    password: string
    role_id?: number | null
    plan_id?: number | null
  }) =>
    apiFetch<AuthUser>('/admin/users', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateUser: (
    userId: number,
    body: {
      is_active?: boolean
      role_id?: number | null
      password?: string
      plan_id?: number | null
    },
  ) =>
    apiFetch<AuthUser>(`/admin/users/${userId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteUser: (userId: number) =>
    apiFetch<void>(`/admin/users/${userId}`, { method: 'DELETE' }),
  cancelUserSubscription: (userId: number, atPeriodEnd = true) =>
    apiFetch<AuthUser>(
      `/admin/users/${userId}/billing/cancel?at_period_end=${atPeriodEnd ? 'true' : 'false'}`,
      { method: 'POST' },
    ),
  resumeUserSubscription: (userId: number) =>
    apiFetch<AuthUser>(`/admin/users/${userId}/billing/resume`, { method: 'POST' }),
  syncUserSubscription: (userId: number) =>
    apiFetch<AuthUser>(`/admin/users/${userId}/billing/sync`, { method: 'POST' }),
  listProjects: (params?: { user_id?: number; pipeline_type?: string }) => {
    const query = new URLSearchParams()
    if (params?.user_id != null) query.set('user_id', String(params.user_id))
    if (params?.pipeline_type) query.set('pipeline_type', params.pipeline_type)
    const suffix = query.toString() ? `?${query}` : ''
    return apiFetch<ProjectSummary[]>(`/admin/projects${suffix}`)
  },
  getProject: (projectId: number) =>
    apiFetch<ProjectDetail>(`/admin/projects/${projectId}`),
  downloadProjectArtifact: (projectId: number, filename: string) =>
    projectArtifactUrl(projectId, filename, 'admin'),
  updateProject: (
    projectId: number,
    body: { title?: string; status?: string },
  ) =>
    apiFetch<ProjectDetail>(`/admin/projects/${projectId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteProject: (projectId: number) =>
    apiFetch<void>(`/admin/projects/${projectId}`, { method: 'DELETE' }),
  listPlantModelConversations: (params?: { user_id?: number; status?: string }) => {
    const query = new URLSearchParams()
    if (params?.user_id != null) query.set('user_id', String(params.user_id))
    if (params?.status) query.set('status', params.status)
    const suffix = query.toString() ? `?${query}` : ''
    return apiFetch<PlantModelConversationSummary[]>(
      `/admin/plant-model/conversations${suffix}`,
    )
  },
  getPlantModelConversation: (conversationId: number) =>
    apiFetch<PlantModelConversationDetail>(
      `/admin/plant-model/conversations/${conversationId}`,
    ),
  deletePlantModelConversation: (conversationId: number) =>
    apiFetch<void>(`/admin/plant-model/conversations/${conversationId}`, {
      method: 'DELETE',
    }),
  getApiKeys: () => apiFetch<ApiKeysResponse>('/admin/api-keys'),
  updateApiKeys: (body: ApiKeysUpdate) =>
    apiFetch<ApiKeysResponse>('/admin/api-keys', {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  listSsoProviders: () => apiFetch<SsoProviderAdmin[]>('/admin/sso-providers'),
  createSsoProvider: (body: SsoProviderCreate) =>
    apiFetch<SsoProviderAdmin>('/admin/sso-providers', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateSsoProvider: (providerId: number, body: SsoProviderUpdate) =>
    apiFetch<SsoProviderAdmin>(`/admin/sso-providers/${providerId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteSsoProvider: (providerId: number) =>
    apiFetch<void>(`/admin/sso-providers/${providerId}`, { method: 'DELETE' }),
  getErrorTrackingSettings: () =>
    apiFetch<ErrorTrackingSettings>('/admin/errors/settings'),
  updateErrorTrackingSettings: (body: Partial<ErrorTrackingSettings>) =>
    apiFetch<ErrorTrackingSettings>('/admin/errors/settings', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  listErrors: (params?: {
    user_id?: number
    source?: string
    status_code?: number
    q?: string
    limit?: number
  }) =>
    apiFetch<ErrorEvent[]>(
      `/admin/errors${buildQuery({
        user_id: params?.user_id,
        source: params?.source,
        status_code: params?.status_code,
        q: params?.q,
        limit: params?.limit,
      })}`,
    ),
  downloadErrorsCsv: (params?: {
    user_id?: number
    source?: string
    status_code?: number
    q?: string
    limit?: number
  }) =>
    downloadAdminCsv('/admin/errors/export.csv', {
      user_id: params?.user_id,
      source: params?.source,
      status_code: params?.status_code,
      q: params?.q,
      limit: params?.limit,
    }),
  listAuditLog: (params?: {
    category?: string
    action?: string
    actor_user_id?: number
    success?: boolean
    q?: string
    limit?: number
  }) =>
    apiFetch<AuditLogEntry[]>(
      `/admin/audit-log${buildQuery({
        category: params?.category,
        action: params?.action,
        actor_user_id: params?.actor_user_id,
        success: params?.success,
        q: params?.q,
        limit: params?.limit,
      })}`,
    ),
  downloadAuditLogCsv: (params?: {
    category?: string
    action?: string
    actor_user_id?: number
    success?: boolean
    q?: string
    limit?: number
  }) =>
    downloadAdminCsv('/admin/audit-log/export.csv', {
      category: params?.category,
      action: params?.action,
      actor_user_id: params?.actor_user_id,
      success: params?.success,
      q: params?.q,
      limit: params?.limit,
    }),
  downloadUsersCsv: () => downloadAdminCsv('/admin/users/export.csv'),
  downloadPlansCsv: () => downloadAdminCsv('/admin/plans/export.csv'),
  downloadProjectsCsv: (params?: { user_id?: number; pipeline_type?: string }) =>
    downloadAdminCsv('/admin/projects/export.csv', {
      user_id: params?.user_id,
      pipeline_type: params?.pipeline_type,
    }),
  downloadProjectsProfilingCsv: (params?: { user_id?: number; pipeline_type?: string }) =>
    downloadAdminCsv('/admin/projects/profiling/export.csv', {
      user_id: params?.user_id,
      pipeline_type: params?.pipeline_type,
    }),
  downloadMonitoringCsv: () => downloadAdminCsv('/admin/monitoring/export.csv'),
  downloadOverviewCsv: () => downloadAdminCsv('/admin/overview/export.xlsx'),
  downloadProfileSurveyCsv: () => downloadAdminCsv('/admin/survey/responses/profile/export.csv'),
  downloadBeforeTestSurveyCsv: () => downloadAdminCsv('/admin/survey/responses/before-test/export.csv'),
  downloadFeedbackSurveyCsv: () => downloadAdminCsv('/admin/survey/responses/feedback/export.csv'),
  getSurveySettings: () => apiFetch<SurveySettings>('/admin/survey/settings'),
  updateSurveySettings: (body: Partial<SurveySettings>) =>
    apiFetch<SurveySettings>('/admin/survey/settings', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  listSurveyResponses: () => apiFetch<SurveyResponses>('/admin/survey/responses'),
}

export const tutorialsApi = {
  listVideos: () => apiFetch<TutorialVideo[]>('/tutorials/videos'),
  listDocuments: () => apiFetch<TutorialDocumentSummary[]>('/tutorials/documents'),
  getDocument: (slug: string) =>
    apiFetch<TutorialDocument>(`/tutorials/documents/${encodeURIComponent(slug)}`),
  listTemplates: () => apiFetch<ControlDesignTemplate[]>('/tutorials/templates'),
}

export const adminTutorialsApi = {
  listVideos: () => apiFetch<TutorialVideo[]>('/admin/tutorials/videos'),
  uploadVideo: (title: string, file: File) => {
    const form = new FormData()
    form.append('title', title)
    form.append('file', file)
    return apiFetch<TutorialVideo>('/admin/tutorials/videos', {
      method: 'POST',
      body: form,
    })
  },
  updateVideo: (videoId: number, body: { title?: string; sort_order?: number }) =>
    apiFetch<TutorialVideo>(`/admin/tutorials/videos/${videoId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteVideo: (videoId: number) =>
    apiFetch<void>(`/admin/tutorials/videos/${videoId}`, { method: 'DELETE' }),

  listDocuments: () => apiFetch<TutorialDocument[]>('/admin/tutorials/documents'),
  getDocument: (documentId: number) =>
    apiFetch<TutorialDocument>(`/admin/tutorials/documents/${documentId}`),
  createDocument: (body: {
    title: string
    slug?: string | null
    body_markdown?: string
    sort_order?: number | null
  }) =>
    apiFetch<TutorialDocument>('/admin/tutorials/documents', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateDocument: (
    documentId: number,
    body: {
      title?: string
      slug?: string
      body_markdown?: string
      sort_order?: number
    },
  ) =>
    apiFetch<TutorialDocument>(`/admin/tutorials/documents/${documentId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteDocument: (documentId: number) =>
    apiFetch<void>(`/admin/tutorials/documents/${documentId}`, { method: 'DELETE' }),

  listTemplates: () => apiFetch<ControlDesignTemplate[]>('/admin/tutorials/templates'),
  uploadTemplate: (title: string, description: string, file: File) => {
    const form = new FormData()
    form.append('title', title)
    form.append('description', description)
    form.append('file', file)
    return apiFetch<ControlDesignTemplate>('/admin/tutorials/templates', {
      method: 'POST',
      body: form,
    })
  },
  updateTemplate: (
    templateId: number,
    body: { title?: string; description?: string; sort_order?: number },
  ) =>
    apiFetch<ControlDesignTemplate>(`/admin/tutorials/templates/${templateId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteTemplate: (templateId: number) =>
    apiFetch<void>(`/admin/tutorials/templates/${templateId}`, { method: 'DELETE' }),
}

export const surveyApi = {
  status: () => apiFetch<SurveyStatus>('/survey/status', {}, AUTH_TIMEOUT_MS),
  submitProfile: (body: ProfileSurveyRequest) =>
    apiFetch<AuthUser>('/survey/profile', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  submitFeedback: (body: OutroSurveyPayload | FeedbackSurveyRequest) =>
    apiFetch<FeedbackSurveyResponseRow>('/survey/feedback', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  submitBeforeTest: (body: BeforeTestSurveyPayload) =>
    apiFetch<BeforeTestSurveyResponseRow>('/survey/before-test', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  dismissTutorial: (action: 'remind_later' | 'dont_show_again') =>
    apiFetch<SurveyStatus>('/survey/tutorial/dismiss', {
      method: 'POST',
      body: JSON.stringify({ action }),
    }),
}

export const projectsApi = {
  list: () => apiFetch<ProjectSummary[]>('/projects'),
  get: (projectId: number) => apiFetch<ProjectDetail>(`/projects/${projectId}`),
  create: (body: {
    title?: string
    pipeline_type: ProjectPipelineType
    file_name?: string
    file_type?: string
    file_content?: string
    llm_model?: string
    control_objective?: string
  }) =>
    apiFetch<ProjectDetail>('/projects', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  update: (
    projectId: number,
    body: {
      title?: string
      status?: string
      control_objective?: string
      file_name?: string
      file_type?: string
      file_content?: string
      job_id?: string
      results?: Record<string, unknown>
    },
  ) =>
    apiFetch<ProjectDetail>(`/projects/${projectId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  downloadArtifact: (projectId: number, filename: string) =>
    projectArtifactUrl(projectId, filename, 'user'),

  simulateSilo: (
    projectId: number,
    body: { gains: Record<string, number>; scenario?: Record<string, unknown> },
  ) =>
    apiFetch<SiloSimulateResponse>(`/projects/${projectId}/silo/simulate`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  submitGrade: (projectId: number, body: { rating: number; comment?: string | null }) =>
    apiFetch<{ status: string; rating: number }>(`/projects/${projectId}/grade`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}

export const healthApi = {
  check: () => apiFetch<{ status: string }>('/health'),
  models: () => apiFetch<ModelsResponse>('/models'),
}

export const uploadApi = {
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return apiFetch<UploadResponse>('/upload', { method: 'POST', body: form })
  },
}

export const bugReportsApi = {
  status: () => apiFetch<BugReportSettings>('/bug-reports/status'),
  create: (body: { description: string; page_url?: string; image?: File | null }) => {
    const form = new FormData()
    form.append('description', body.description)
    if (body.page_url) form.append('page_url', body.page_url)
    if (body.image) form.append('image', body.image)
    return apiFetch<BugReport>('/bug-reports', { method: 'POST', body: form })
  },
  getSettings: () => apiFetch<BugReportSettings>('/admin/bug-reports/settings'),
  updateSettings: (body: Partial<BugReportSettings>) =>
    apiFetch<BugReportSettings>('/admin/bug-reports/settings', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  listAdmin: (params?: { status?: 'open' | 'fixed' | 'all' }) =>
    apiFetch<BugReport[]>(
      `/admin/bug-reports${buildQuery({ status: params?.status })}`,
    ),
  getAdmin: (reportId: number) => apiFetch<BugReport>(`/admin/bug-reports/${reportId}`),
  updateStatus: (reportId: number, status: 'open' | 'fixed') =>
    apiFetch<BugReport>(`/admin/bug-reports/${reportId}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    }),
  downloadCsv: (params?: { status?: 'open' | 'fixed' | 'all' }) =>
    downloadAdminCsv('/admin/bug-reports/export.csv', {
      status: params?.status,
    }),
}

export const plantModelApi = {
  chat: (body: {
    messages: PlantModelChatMessage[]
    user_message: string
    model?: string
    session_state?: PlantModelSessionState | null
    conversation_id?: number | null
    max_drafts?: number
    min_user_turns_before_completion?: number
    web_search_enabled?: boolean
  }) =>
    apiFetch<PlantModelChatResponse>('/plant-model/chat', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  listConversations: () =>
    apiFetch<PlantModelConversationSummary[]>('/plant-model/conversations'),
  getConversation: (conversationId: number) =>
    apiFetch<PlantModelConversationDetail>(`/plant-model/conversations/${conversationId}`),
  deleteConversation: (conversationId: number) =>
    apiFetch<void>(`/plant-model/conversations/${conversationId}`, { method: 'DELETE' }),
  upload: async (file: File) => {
    const form = new FormData()
    form.append('file', file)
    // apiFetch assumes JSON; use raw fetch for multipart
    const base = (import.meta as any).env?.VITE_API_BASE || '/api/v1'
    const res = await fetch(`${base}/plant-model/upload`, { method: 'POST', body: form })
    if (!res.ok) {
      let detail = res.statusText
      try {
        const body = await res.json()
        if (body?.detail) detail = String(body.detail)
      } catch { /* ignore */ }
      throw new Error(detail)
    }
    return res.json() as Promise<import('./types').UploadResponse>
  },
  simulate: (body: import('./types').SimulateRequest) =>
    apiFetch<import('./types').SimulateResponse>('/plant-model/simulate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}

export const regularizerApi = {
  regularize: (body: {
    file_content: string
    file_name?: string
    file_type?: string
    model?: string
  }) =>
    apiFetch<RegularizeResponse>('/regularize', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  standardize: (body: {
    file_content: string
    model?: string
    silo_pipeline?: boolean
  }) =>
    apiFetch<StandardizeResponse>('/regularize/standardize', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}

export const recommenderApi = {
  start: (body: {
    file_content: string
    file_name: string
    model?: string
    step?: string
    user_prompt?: string
  }) =>
    apiFetch<JobResponse>('/recommender/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  state: (jobId: string) =>
    apiFetch<Record<string, unknown>>(`/recommender/${jobId}/state`),

  ragDecision: (jobId: string, body: { flags: string[]; model?: string }) =>
    apiFetch<JobResponse>(`/recommender/${jobId}/rag-decision`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  ragStatus: (jobId: string) =>
    apiFetch<RagStatusResponse>(`/recommender/${jobId}/rag-status`),

  handoff: (jobId: string, body: { chosen_controller?: string | null }) =>
    apiFetch<RecommenderHandoffResponse>(`/recommender/${jobId}/handoff`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}

export const trimmerApi = {
  start: (body: {
    file_content: string
    file_name: string
    model?: string
    trimming_params?: Record<string, unknown>
    states_inputs?: string[]
    project_id?: number | null
    recommender_job_id?: string | null
  }) =>
    apiFetch<JobResponse>('/trimmer/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  input: (jobId: string, body: { key: string; prompt: string; answer: string }) =>
    apiFetch<JobResponse>(`/trimmer/${jobId}/input`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  artifacts: (jobId: string) =>
    apiFetch<TrimmerArtifactsResponse>(`/trimmer/${jobId}/artifacts`),

  timeResponse: (jobId: string) =>
    apiFetch<{ filename: string; message: string }>(`/trimmer/${jobId}/time-response`, {
      method: 'POST',
    }),

  generatePdf: (jobId: string, body?: { recommender_job_id?: string | null }) =>
    apiFetch<{ filename: string; message: string }>(`/trimmer/${jobId}/pdf`, {
      method: 'POST',
      body: JSON.stringify(body ?? {}),
    }),
}

export const siloApi = {
  start: (body: {
    config: Record<string, unknown>
    control_objective?: string
    project_id?: number | null
  }) =>
    apiFetch<JobResponse>('/silo/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  monitor: (jobId: string) =>
    apiFetch<Record<string, unknown>>(`/silo/${jobId}/monitor`),

  simulate: (
    jobId: string,
    body: { gains: Record<string, number>; scenario?: Record<string, unknown> },
  ) =>
    apiFetch<SiloSimulateResponse>(`/silo/${jobId}/simulate`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}

export const muloApi = {
  init: (body: {
    run_config: Record<string, unknown>
    controller_structure: Record<string, unknown>[]
    system_identification: Record<string, unknown>
    trimming_result: Record<string, unknown>
    equation: string
    project_id?: number | null
    file_name?: string
    file_type?: string
  }) =>
    apiFetch<JobResponse>('/mulo/init', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  start: (body: {
    run_config: Record<string, unknown>
    controller_structure: Record<string, unknown>[]
    system_identification: Record<string, unknown>
    trimming_result: Record<string, unknown>
    equation: string
    project_id?: number | null
    file_name?: string
    file_type?: string
  }) =>
    apiFetch<JobResponse>('/mulo/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  configure: (
    jobId: string,
    body: { case_study: Record<string, unknown>; controller_structure: Record<string, unknown>[] },
  ) =>
    apiFetch<JobResponse>(`/mulo/${jobId}/configure`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  run: (jobId: string) =>
    apiFetch<JobResponse>(`/mulo/${jobId}/run`, {
      method: 'POST',
    }),

  continue: (
    jobId: string,
    body: { equation: string; controller_structure: Record<string, unknown>[] },
  ) =>
    apiFetch<JobResponse>(`/mulo/${jobId}/continue`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  state: (jobId: string) => apiFetch<MuloDesignerStateResponse>(`/mulo/${jobId}/state`),

  simulate: (
    jobId: string,
    body: {
      kp: number
      ki: number
      kd: number
      signal_type: string
      amplitude?: number
    },
  ) =>
    apiFetch<MuloSimulateResponse>(`/mulo/${jobId}/simulate`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  scratchpad: (
    jobId: string,
    body: { modified_code: string; modified_controller_structure: Record<string, unknown>[] },
  ) =>
    apiFetch<JobResponse>(`/mulo/${jobId}/scratchpad`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  plotData: (jobId: string) =>
    apiFetch<Record<string, unknown>>(`/mulo/${jobId}/plot-data`),
}

export const jobsApi = {
  status: (jobId: string) => apiFetch<JobStatusResponse>(`/jobs/${jobId}`),
  cancel: (jobId: string) =>
    apiFetch<JobStatusResponse>(`/jobs/${jobId}/cancel`, { method: 'POST' }),
  results: (jobId: string) => apiFetch<ArtifactResponse>(`/jobs/${jobId}/results`),
  downloadArtifact: artifactUrl,
}

export const caseStudiesApi = {
  list: () => apiFetch<CaseStudiesResponse>('/case-studies'),
  mulo: (name: string) =>
    apiFetch<Record<string, unknown>>(`/case-studies/mulo/${encodeURIComponent(name)}`),
}

export const siteApi = {
  getLanding: () => apiFetch<LandingPayload>('/site/landing'),
}

export const blogApi = {
  list: () => apiFetch<BlogPostListItem[]>('/blog'),
  get: (slug: string) => apiFetch<BlogPost>(`/blog/${encodeURIComponent(slug)}`),
}

export const adminSiteApi = {
  getBrand: () => apiFetch<SiteBrand>('/admin/site/brand'),
  updateBrand: (body: SiteBrand) =>
    apiFetch<SiteBrand>('/admin/site/brand', {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  getLanding: () => apiFetch<Record<string, unknown>>('/admin/site/landing'),
  updateLanding: (body: Record<string, unknown>) =>
    apiFetch<Record<string, unknown>>('/admin/site/landing', {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  listMenus: (location?: string) =>
    apiFetch<NavMenuItem[]>(`/admin/site/menus${buildQuery({ location })}`),
  createMenu: (body: Omit<NavMenuItem, 'id'>) =>
    apiFetch<NavMenuItem>('/admin/site/menus', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateMenu: (menuId: number, body: Partial<Omit<NavMenuItem, 'id'>>) =>
    apiFetch<NavMenuItem>(`/admin/site/menus/${menuId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteMenu: (menuId: number) =>
    apiFetch<void>(`/admin/site/menus/${menuId}`, { method: 'DELETE' }),
}

export const adminMediaApi = {
  upload: (file: File, prefix = 'image') => {
    const form = new FormData()
    form.append('file', file)
    form.append('prefix', prefix)
    return apiFetch<MediaUploadResponse>('/admin/media', { method: 'POST', body: form })
  },
}

export const adminBlogApi = {
  list: () => apiFetch<BlogPostListItem[]>('/admin/blog'),
  get: (postId: number) => apiFetch<BlogPost>(`/admin/blog/${postId}`),
  create: (body: {
    title: string
    slug?: string | null
    excerpt?: string
    body_markdown?: string
    cover_image_url?: string | null
    status?: string
  }) =>
    apiFetch<BlogPost>('/admin/blog', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  update: (
    postId: number,
    body: {
      title?: string
      slug?: string
      excerpt?: string
      body_markdown?: string
      cover_image_url?: string | null
      status?: string
    },
  ) =>
    apiFetch<BlogPost>(`/admin/blog/${postId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  delete: (postId: number) =>
    apiFetch<void>(`/admin/blog/${postId}`, { method: 'DELETE' }),
}

export const adaptiveApi = {
  createJob: (body: AdaptiveJobCreateRequest) =>
    apiFetch<AdaptiveJobCreateResponse>('/adaptive/jobs', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  listJobs: () => apiFetch<AdaptiveJobSummary[]>('/adaptive/jobs'),
  getJob: (jobId: string) => apiFetch<AdaptiveJobStatusResponse>(`/adaptive/jobs/${jobId}`),
  clarify: (jobId: string, body: AdaptiveClarifyRequest) =>
    apiFetch<AdaptiveClarifyResponse>(`/adaptive/jobs/${jobId}/clarify`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  cancel: (jobId: string) =>
    apiFetch<AdaptiveJobStatusResponse>(`/adaptive/jobs/${jobId}/cancel`, { method: 'POST' }),
  getResults: (jobId: string) =>
    apiFetch<AdaptiveJobResultsResponse>(`/adaptive/jobs/${jobId}/results`),
  diagnosisChat: (jobId: string, body: AdaptiveDiagnosisChatRequest) =>
    apiFetch<AdaptiveDiagnosisChatResponse>(`/adaptive/jobs/${jobId}/diagnosis/chat`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  streamEvents: (
    jobId: string,
    onEvent: (event: string, data: any) => void,
    onError?: (err: unknown) => void,
  ) => streamEvents(`/adaptive/jobs/${jobId}/events`, onEvent, onError),
  getExportScriptUrl: (jobId: string) => `${API_BASE}/adaptive/jobs/${jobId}/export-script`,
  getExportScript: (jobId: string) =>
    apiFetch<string>(`/adaptive/jobs/${jobId}/export-script`),
  submitGrade: (jobId: string, body: { rating: number; comment?: string | null }) =>
    apiFetch<{ status: string; rating: number }>(`/adaptive/jobs/${jobId}/grade`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getReportPdfUrl: (jobId: string) => {
    const token = getAuthToken()
    return `${API_BASE}/adaptive/jobs/${jobId}/report.pdf${token ? `?access_token=${encodeURIComponent(token)}` : ''}`
  },
  /** Authenticated PDF download — browser navigation does not send JWT. */
  downloadReportPdf: async (jobId: string, filename?: string): Promise<void> => {
    const token = getAuthToken()
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 45000)
    try {
      const response = await fetch(`${API_BASE}/adaptive/jobs/${jobId}/report.pdf`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        signal: controller.signal,
      })
      if (!response.ok) {
        let detail = `Failed to download PDF (${response.status})`
        try {
          const body = await response.json()
          if (body?.detail) detail = String(body.detail)
        } catch {
          /* ignore non-JSON */
        }
        throw new Error(detail)
      }
      const blob = await response.blob()
      const disposition = response.headers.get('Content-Disposition') || ''
      const match = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition)
      const name =
        filename ||
        (match ? match[1].replace(/['"]/g, '') : `adaptive_${jobId}_report.pdf`)
      triggerBlobDownload(blob, name)
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        throw new Error('PDF generation/download timed out after 45 seconds. Please try again.')
      }
      throw err
    } finally {
      clearTimeout(timeoutId)
    }
  },
}

export const mpcApi = {
  createJob: (body: MPCJobCreateRequest) =>
    apiFetch<MPCJobCreateResponse>('/mpc/jobs', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  listJobs: () => apiFetch<MPCJobSummary[]>('/mpc/jobs'),
  getJob: (jobId: string) => apiFetch<MPCJobStatusResponse>(`/mpc/jobs/${jobId}`),
  cancel: (jobId: string) =>
    apiFetch<MPCJobStatusResponse>(`/mpc/jobs/${jobId}/cancel`, { method: 'POST' }),
  getResults: (jobId: string) =>
    apiFetch<MPCJobResultsResponse>(`/mpc/jobs/${jobId}/results`),
  diagnosisChat: (jobId: string, body: AdaptiveDiagnosisChatRequest) =>
    apiFetch<AdaptiveDiagnosisChatResponse>(`/mpc/jobs/${jobId}/diagnosis/chat`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  streamEvents: (
    jobId: string,
    onEvent: (event: string, data: any) => void,
    onError?: (err: unknown) => void,
  ) => streamEvents(`/mpc/jobs/${jobId}/events`, onEvent, onError),
  submitGrade: (jobId: string, body: { rating: number; comment?: string | null }) =>
    apiFetch<{ status: string; rating: number }>(`/mpc/jobs/${jobId}/grade`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  testDynamics: (body: MPCDiagnosticsRequest) =>
    apiFetch<MPCDiagnosticsResponse>('/mpc/test-dynamics', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  simulate: (body: MPCSimulateRequest) =>
    apiFetch<MPCSimulateResponse>('/mpc/simulate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getExportScriptUrl: (jobId: string) => `${API_BASE}/mpc/jobs/${jobId}/export-script`,
  getReportPdfUrl: (jobId: string) => {
    const token = getAuthToken()
    return `${API_BASE}/mpc/jobs/${jobId}/report.pdf${token ? `?access_token=${encodeURIComponent(token)}` : ''}`
  },
  getExportScript: (jobId: string) =>
    apiFetch<string>(`/mpc/jobs/${jobId}/export-script`),
  /** Authenticated PDF download — browser navigation does not send JWT. */
  downloadReportPdf: async (jobId: string, filename?: string): Promise<void> => {
    const token = getAuthToken()
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 45000)
    try {
      const response = await fetch(`${API_BASE}/mpc/jobs/${jobId}/report.pdf`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        signal: controller.signal,
      })
      if (!response.ok) {
        let detail = `Failed to download PDF (${response.status})`
        try {
          const body = await response.json()
          if (body?.detail) detail = String(body.detail)
        } catch {
          /* ignore non-JSON */
        }
        throw new Error(detail)
      }
      const blob = await response.blob()
      const disposition = response.headers.get('Content-Disposition') || ''
      const match = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition)
      const name =
        filename ||
        (match ? match[1].replace(/['"]/g, '') : `mpc_${jobId}_report.pdf`)
      triggerBlobDownload(blob, name)
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        throw new Error('PDF generation/download timed out after 45 seconds. Please try again.')
      }
      throw err
    } finally {
      clearTimeout(timeoutId)
    }
  },
}

export const plantArtifactApi = {
  createArtifact: (body: ArtifactCreateRequest) =>
    apiFetch<ArtifactCreateResponse>('/plant-model/artifacts', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  listArtifacts: () => apiFetch<ArtifactSummary[]>('/plant-model/artifacts'),
  getArtifact: (artifactId: string) =>
    apiFetch<ArtifactDetail>(`/plant-model/artifacts/${artifactId}`),
  getPlugin: (artifactId: string) =>
    apiFetch<ArtifactPluginResponse>(`/plant-model/artifacts/${artifactId}/plugin`),
  getAdaptiveSpec: (artifactId: string) =>
    apiFetch<Record<string, unknown>>(`/plant-model/artifacts/${artifactId}/adaptive-spec`),
  validate: (body: ValidationRequest) =>
    apiFetch<ValidationResponse>('/plant-model/validate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}

export const creditsApi = {
  getDashboard: () => apiFetch<CreditDashboard>('/credits/me'),
  getLedger: (limit = 50, offset = 0) =>
    apiFetch<CreditLedgerEntry[]>(`/credits/me/ledger?limit=${limit}&offset=${offset}`),
  getSessions: (limit = 50, offset = 0) =>
    apiFetch<CreditUsageSession[]>(`/credits/me/sessions?limit=${limit}&offset=${offset}`),
  getSession: (sessionId: number) =>
    apiFetch<CreditUsageSession>(`/credits/sessions/${sessionId}`),
}

export const billingApi = {
  getPlans: () => apiFetch<PublicPlan[]>('/billing/plans'),
  getStatus: () => apiFetch<BillingStatus>('/billing/status'),
  createCheckoutSession: (body: { plan_id: number; interval: 'month' | 'year' }) =>
    apiFetch<{ url: string }>('/billing/checkout-session', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  createPortalSession: () =>
    apiFetch<{ url: string }>('/billing/portal-session', { method: 'POST' }),
}


