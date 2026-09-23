export interface JobResponse {
  job_id: string
  module: string
  status: string
}

export interface JobStatusResponse {
  job_id: string
  module: string
  status: string
  metadata: Record<string, unknown>
  error?: string
}

export interface UploadResponse {
  file_name: string
  file_type: string
  file_content: string
}

export interface MediaUploadResponse {
  url: string
}

export interface ModelsResponse {
  llm_models: string[]
  rag_models: string[]
}

export interface RegularizeResponse {
  file_content: string
  change_applied: boolean
  human_intervention: boolean
}

export interface StandardizeResponse {
  file_content: string
}

export interface PlantModelChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface PlantModelResult {
  system_name: string
  python_code: string
  metadata?: Record<string, unknown> | null
}

export interface PlantModelSessionState {
  draft_count: number
  latest_draft: PlantModelResult | null
}

export interface PlantModelTokenUsage {
  input_tokens: number
  output_tokens: number
  estimated_cost: number
}

export interface PlantModelChatResponse {
  reply: string
  status: 'continue' | 'draft' | 'complete'
  final_result: PlantModelResult | null
  session_state: PlantModelSessionState
  usage: PlantModelTokenUsage | null
  conversation_id: number | null
}

export interface PlantModelConversationSummary {
  id: number
  title: string
  status: 'active' | 'complete'
  llm_model: string
  system_name: string | null
  user_id?: number | null
  owner_email?: string | null
  created_at: string
  updated_at: string
}

export interface PlantModelConversationDetail {
  id: number
  title: string
  status: 'active' | 'complete'
  llm_model: string
  messages: PlantModelChatMessage[]
  session_state: PlantModelSessionState | null
  final_result: PlantModelResult | null
  user_id?: number | null
  owner_email?: string | null
  created_at: string
  updated_at: string
}

export interface RecommenderHandoffResponse {
  file_content: string
  chosen_controller: string
  trimming_params: string[]
  states_inputs: string[]
}

export interface TrimmerArtifactsResponse {
  result: Record<string, unknown>
  config: Record<string, unknown>
  pdf_file?: string
  safe_system_name: string
  output_dir: string
  time_response_file?: string
}

export interface TrimmerTimeResponseResponse {
  filename: string
  message: string
}

export interface CaseStudiesResponse {
  python: string[]
  matlab: string[]
  ga_json: string[]
  mulo: string[]
  mulo_objectives: Record<string, string>
}

export interface ArtifactResponse {
  job_id: string
  artifacts: Record<string, unknown>
}

export interface RagStatusResponse {
  next_step: 'comparison' | 'review'
  error_message: string
}

export interface MuloDesignerStateResponse {
  job_id: string
  controller_index: number
  controller_designed: boolean
  total_loops: number
  loop_name: string
  is_complete: boolean
  equation: string
  controller_structure: Record<string, unknown>[]
  case_study: Record<string, unknown>
  run_config: Record<string, unknown>
  final_state: Record<string, unknown>
  modified_code: string
  modified_controller_structure: Record<string, unknown>[]
  pid_gains: { Kp: number; Ki: number; Kd: number }
  pid_gain_bounds: { Kp: number; Ki: number; Kd: number }
}

export interface MuloSimulateResponse {
  signal_type: string
  time: number[]
  actual: number[]
  reference: number[]
  y_label: string
  unit: string
  code: string
  amplitude?: number
}

export interface SiloSimTrace {
  metrics: Record<string, number | boolean | null>
  trajectory: number[]
  control_signals: number[]
  errors?: number[]
}

export interface SiloSimulateResponse {
  controller_type: string
  optimal_gains: Record<string, number>
  manual_gains: Record<string, number>
  param_bounds: Record<string, [number, number]>
  target: number
  dt: number
  max_time: number
  time: number[]
  scenario?: Record<string, unknown>
  initial_condition?: number[]
  initial_condition_value?: number
  initial_condition_range?: number[]
  output_channel?: number
  optimal: SiloSimTrace
  manual: SiloSimTrace | null
}

export type PipelineType = 'siloDesign' | 'muloDesign' | 'adaptiveDesign' | 'mpcDesign' | null

export interface StreamEvent {
  type: string
  mode?: string
  content?: unknown
  step?: string
  job_id?: string
  status?: string
  error?: string
  summary?: Record<string, unknown>
  metadata?: Record<string, unknown>
}

export type ThemeMode = 'light' | 'dark' | 'system' | 'theme_of_day'

export interface AuthUser {
  id: number
  email: string
  display_name: string | null
  avatar_url: string | null
  theme: ThemeMode
  is_admin: boolean
  is_active: boolean
  email_verified?: boolean
  plan_id: number | null
  plan_name: string | null
  role_id: number | null
  role_name: string | null
  actions: string[]
  created_at: string
  profile_survey_completed?: boolean
  feedback_survey_completed?: boolean
  feedback_survey_completed_silo?: boolean
  feedback_survey_completed_mulo?: boolean
  tutorial_dont_show_again?: boolean
  onboarding_answers?: OnboardingAnswers | null
  // -- Stripe billing --
  stripe_customer_id?: string | null
  stripe_subscription_id?: string | null
  stripe_subscription_status?: string | null
  stripe_price_id?: string | null
  billing_interval?: 'month' | 'year' | null
  stripe_current_period_end?: string | null
  stripe_cancel_at_period_end?: boolean
}

export interface BillingStatus {
  stripe_enabled: boolean
  publishable_key: string | null
}

export interface PublicPlan {
  id: number
  plan_code: string | null
  name: string
  description: string
  price: number
  price_yearly: number | null
  is_contact_sales: boolean
  is_most_popular: boolean
  purchasable: boolean
  actions: string[]
}

export interface MessageResponse {
  message: string
}

export interface AuthSessionInfo {
  id: number
  ip_address: string | null
  user_agent: string | null
  created_at: string
  last_seen_at: string
  is_current: boolean
}

export interface UserProfileSurveyDetail {
  university: string | null
  degree: string | null
  major: string | null
  matlab_experience: string | null
  control_design_experience: string | null
  completed_at: string | null
}

export interface UserFeedbackSurveyDetail {
  pipeline_type: FeedbackPipelineType
  satisfaction: number
  ease_of_use: number
  product_value: number
  confidence: number
  reuse_intention: number
  willingness_to_pay: number
  main_problems: string
  created_at: string
}

export interface LoginHistoryEntry {
  id: number
  email: string
  success: boolean
  ip_address: string | null
  user_agent: string | null
  failure_reason: string | null
  created_at: string
}

export interface AdminUserDetail {
  user: AuthUser
  allowed_models: string[]
  profile_survey: UserProfileSurveyDetail | null
  feedback_surveys: UserFeedbackSurveyDetail[]
  projects: ProjectSummary[]
  errors: ErrorEvent[]
  login_history: LoginHistoryEntry[]
  sessions?: AuthSessionInfo[]
}

export type JourneyStepStatus = 'ok' | 'fail' | 'info'
export type JourneyHealth = 'good' | 'at_risk' | 'new'
export type JourneyActionPriority = 'high' | 'med' | 'low'

export interface JourneyUser {
  id: number
  email: string
  display_name: string | null
}

export interface JourneyStep {
  id: string
  kind: string
  title: string
  timestamp: string
  status: JourneyStepStatus
  duration_seconds: number | null
  detail: string | null
  error: string | null
  source_id: string | null
}

export interface JourneyComment {
  id: string
  source: string
  text: string
  timestamp: string
  rating: number | null
  meta: string | null
}

export interface JourneySignal {
  key: string
  value: string
  positive: boolean
}

export interface JourneyPersonaAlt {
  label: string
  score: number
}

export interface JourneyPersona {
  label: string
  score: number
  signals: JourneySignal[]
  alts: JourneyPersonaAlt[]
}

export interface JourneyProfile {
  university: string | null
  degree: string | null
  major: string | null
  matlab_experience: string | null
  control_design_experience: string | null
  completed_at: string | null
}

export interface JourneyKpis {
  sessions: number
  sessions_completed: number
  sessions_abandoned: number
  success_rate: number | null
  avg_score: number | null
  best_score: number | null
  worst_score: number | null
  tokens_total: number
  credits_charged: number
  error_count: number
  error_by_source: Record<string, number>
  time_seconds: number
  median_session_seconds: number | null
}

export interface JourneyPipelineMix {
  name: string
  pipeline_type: string
  count: number
  pct: number
}

export interface JourneySession {
  id: number
  title: string
  pipeline: string
  pipeline_type: string
  status: string
  score: number | null
  duration_seconds: number | null
  tokens: number | null
  updated_at: string
}

export interface JourneyUsage {
  tokens_total: number
  credits_charged: number
  avg_credits_per_success: number | null
  api_errors: number
  other_errors: number
  avg_rating: number | null
  rating_count: number
}

export interface JourneyFlags {
  email_verified: boolean
  is_active: boolean
  bug_report_count: number
  profile_survey_complete: boolean
}

export interface JourneyAction {
  priority: JourneyActionPriority
  title: string
  detail: string
}

export interface JourneyDossier {
  joined_at: string | null
  last_active_at: string | null
  plan_name: string | null
  health: JourneyHealth
  health_detail: string
  tags: string[]
  kpis: JourneyKpis
  persona: JourneyPersona | null
  profile: JourneyProfile | null
  pipeline_mix: JourneyPipelineMix[]
  recent_sessions: JourneySession[]
  usage: JourneyUsage
  flags: JourneyFlags
  actions: JourneyAction[]
  latest_project_id: number | null
}

export interface UserJourney {
  user: JourneyUser
  steps: JourneyStep[]
  comments: JourneyComment[]
  dossier: JourneyDossier
}

export type ExperienceLevel = 'None' | 'Beginner' | 'Intermediate' | 'Advanced'
export type DegreeLevel = "Bachelor's" | "Master's" | 'PhD' | 'Other'
export type FeedbackPipelineType = 'siloDesign' | 'muloDesign' | 'adaptiveDesign' | 'mpcDesign'
export type MajorField =
  | 'Electrical Engineering'
  | 'Mechanical Engineering'
  | 'Chemical Engineering'
  | 'Aerospace Engineering'
  | 'Computer Science'
  | 'Control Engineering'
  | 'Mechatronics'
  | 'Other'

export interface SurveySettings {
  enabled: boolean
}

export interface TutorialVideo {
  id: number
  title: string
  file_url: string
  sort_order: number
  created_at: string
}

export interface TutorialDocumentSummary {
  id: number
  title: string
  slug: string
  sort_order: number
  created_at: string
  updated_at: string
}

export interface TutorialDocument extends TutorialDocumentSummary {
  body_markdown: string
}

export interface ControlDesignTemplate {
  id: number
  title: string
  description: string
  file_url: string
  original_filename: string
  sort_order: number
  created_at: string
}

export interface SurveyStatus {
  enabled: boolean
  needs_profile_survey: boolean
  before_test_completed?: boolean
  feedback_completed: boolean
  feedback_completed_silo: boolean
  feedback_completed_mulo: boolean
  show_tutorial: boolean
  videos: TutorialVideo[]
}

export interface OnboardingAnswers {
  tools: string[]
  experience: string | null
  recency?: string | null
  goals: string[]
  other_text?: string
  pipeline: string | null
  completed?: boolean
  skipped?: boolean
}

export interface ProfileSurveyRequest {
  tools?: string[]
  experience?: string | null
  recency?: string | null
  goals?: string[]
  other_text?: string
  pipeline?: string | null
  completed?: boolean
  skipped?: boolean
  onboarding_answers?: OnboardingAnswers

  university?: string
  degree?: DegreeLevel | string
  major?: MajorField | string
  matlab_experience?: ExperienceLevel | string
  control_design_experience?: ExperienceLevel | string
}

export interface FeedbackSurveyRequest {
  pipeline_type: FeedbackPipelineType
  satisfaction: number
  ease_of_use: number
  product_value: number
  confidence: number
  reuse_intention: number
  willingness_to_pay: number
  main_problems: string
}

export interface ProfileSurveyResponseRow {
  user_id: number
  email: string
  university: string | null
  degree: string | null
  major: string | null
  matlab_experience: string | null
  control_design_experience: string | null
  onboarding_answers?: OnboardingAnswers | null
  completed_at: string | null
}

export interface FeedbackSurveyResponseRow {
  id?: number
  user_id: number
  email: string
  pipeline_type: FeedbackPipelineType
  job_id?: string | null
  project_id?: number | null
  plant_name?: string | null
  score?: number | null
  success?: boolean | null
  technical_usefulness?: number | null
  technical_usefulness_na?: boolean
  trust?: number | null
  trust_na?: boolean
  satisfaction?: number
  ease_of_use: number
  product_value?: number
  confidence?: number
  reuse_intention: number
  willingness_to_pay?: number
  nps?: number | null
  main_problems: string
  is_bug?: boolean
  created_at: string
}

export interface BeforeTestSurveyResponseRow {
  id: number
  user_id: number
  email: string
  q1_last_worked: string
  q2_time_spent: string
  q3_knowledge_gaps: number
  q4_difficult_parts: string[]
  q5_biggest_problem: string
  q6_help_sources: string[]
  q7_considered_paying: string
  q8a_amount_hired?: string | null
  q8b_amount_paid_to_user?: string | null
  q9_impact: string[]
  created_at: string
}

export interface SurveyResponses {
  profile: ProfileSurveyResponseRow[]
  feedback: FeedbackSurveyResponseRow[]
  before_test?: BeforeTestSurveyResponseRow[]
}

export interface OutroSurveyPayload {
  pipeline_type: FeedbackPipelineType
  job_id?: string | null
  project_id?: number | null
  plant_name?: string | null
  score?: number | null
  success?: boolean | null
  technical_usefulness?: number | null
  technical_usefulness_na?: boolean
  trust?: number | null
  trust_na?: boolean
  ease_of_use: number
  reuse_intention: number
  nps?: number | null
  main_problems?: string
  is_bug?: boolean
  satisfaction?: number
  confidence?: number
}

export interface BeforeTestSurveyPayload {
  q1_last_worked: string
  q2_time_spent: string
  q3_knowledge_gaps: number
  q4_difficult_parts: string[]
  q5_biggest_problem: string
  q6_help_sources: string[]
  q7_considered_paying: string
  q8a_amount_hired?: string | null
  q8b_amount_paid_to_user?: string | null
  q9_impact: string[]
}

export interface ActionInfo {
  code: string
  description: string
}

export interface PlanInfo {
  id: number
  name: string
  description: string
  price: number
  is_active: boolean
  actions: string[]
  models: string[]
  created_at: string
  // -- Stripe billing --
  plan_code?: string | null
  price_yearly?: number | null
  is_contact_sales?: boolean
  is_most_popular?: boolean
  stripe_product_id?: string | null
  stripe_price_id_monthly?: string | null
  stripe_price_id_yearly?: string | null
  stripe_configured?: boolean
}

export interface RoleInfo {
  id: number
  name: string
  description: string
  is_system: boolean
  is_active: boolean
  actions: string[]
  created_at: string
}

export interface DefaultPlanInfo {
  plan_id: number | null
  plan: PlanInfo | null
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export type ProjectPipelineType = 'siloDesign' | 'muloDesign' | 'adaptiveDesign' | 'mpcDesign'
export type ProjectStatus = 'draft' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface SessionMetadata {
  tokens?: {
    prompt?: number
    completion?: number
    total?: number
    total_tokens?: number
    [key: string]: unknown
  } | null
  cost_usd?: number | null
  error_counts?: number | null
  wall_clock_time_s?: number | null
  wall_clock_time_seconds?: number | null
  [key: string]: unknown
}

export interface DesignGradePayload {
  rating: number
  comment?: string | null
}

export interface ProjectSummary {
  id: number
  user_id: number
  owner_email?: string | null
  title: string
  pipeline_type: ProjectPipelineType
  status: ProjectStatus
  file_name: string
  file_type: string
  file_url?: string | null
  llm_model: string
  has_results: boolean
  job_id?: string | null
  score?: number | null
  success?: boolean | null
  rating?: number | null
  design_grade?: { rating: number; comment?: string | null; created_at?: string | null } | null
  session_metadata?: SessionMetadata | null
  created_at: string
  updated_at: string
}

export interface ProjectDetail extends ProjectSummary {
  file_content: string
  control_objective?: string | null
  results?: Record<string, unknown> | null
}

export interface MemoryMetrics {
  used_bytes: number
  total_bytes: number
  percent: number
}

export interface DiskMetrics {
  used_bytes: number
  total_bytes: number
  percent: number
}

export interface NetworkMetrics {
  bytes_sent: number
  bytes_recv: number
  sent_rate_bps: number
  recv_rate_bps: number
}

export interface ApiMetrics {
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  error_rate_percent: number
  requests_in_window: number
}

export interface MonitoringSnapshot {
  collected_at: string
  uptime_seconds: number
  cpu_percent: number
  memory: MemoryMetrics
  disk: DiskMetrics
  network: NetworkMetrics
  api: ApiMetrics
}

export interface MonitoringResponse {
  current: MonitoringSnapshot
  history: MonitoringSnapshot[]
}

export interface AnalyticsSeriesPoint {
  date: string
  count: number
}

export interface AnalyticsModuleCount {
  module: string
  count: number
}

export interface AnalyticsLlmCount {
  model: string
  count: number
}

export interface AnalyticsResponse {
  days: number
  dau_today: number
  mau: number
  retention_d7: number | null
  retention_d30: number | null
  dau_series: AnalyticsSeriesPoint[]
  mau_series: AnalyticsSeriesPoint[]
  modules: AnalyticsModuleCount[]
  llms: AnalyticsLlmCount[]
  most_used_llm: string | null
}

export interface TelegramAnalyticsSettings {
  enabled: boolean
  chat_id: string
  send_hour_utc: number
  bot_token_configured: boolean
  bot_token_masked: string
  last_sent_date: string | null
}

export interface TelegramAnalyticsSettingsUpdate {
  enabled?: boolean
  chat_id?: string
  send_hour_utc?: number
  /** Write-only; omit to leave unchanged; empty string clears. */
  bot_token?: string
}

export interface TelegramAnalyticsTestResult {
  ok: boolean
  detail: string
}

export interface ErrorTrackingSettings {
  enabled: boolean
  frontend: boolean
  backend: boolean
  api: boolean
}

export type ApiKeyName =
  | 'OPENAI_API_KEY'
  | 'NVIDIA_API_KEY'
  | 'GROQ_API_KEY'
  | 'CEREBRAS_API_KEY'
  | 'TAVILY_API_KEY'
  | 'STRIPE_SECRET_KEY'
  | 'STRIPE_PUBLISHABLE_KEY'
  | 'STRIPE_WEBHOOK_SECRET'

export interface ApiKeyStatus {
  name: ApiKeyName | string
  configured: boolean
  masked_value: string
}

export interface ApiKeysResponse {
  keys: ApiKeyStatus[]
}

export type ApiKeysUpdate = Partial<Record<ApiKeyName, string>>

export type SsoProviderKey = 'google' | 'github'

export interface SsoProviderPublic {
  id: number
  provider: SsoProviderKey
  display_name: string
}

export interface SsoProviderAdmin {
  id: number
  provider: SsoProviderKey
  display_name: string
  client_id: string
  client_secret_configured: boolean
  client_secret_masked: string
  enabled: boolean
  created_at: string
  updated_at: string
}

export interface SsoProviderCreate {
  provider: SsoProviderKey
  display_name: string
  client_id: string
  client_secret: string
  enabled?: boolean
}

export interface SsoProviderUpdate {
  display_name?: string
  client_id?: string
  client_secret?: string
  enabled?: boolean
}

export type ErrorEventSource = 'frontend' | 'backend' | 'api'

export interface ErrorEvent {
  id: number
  source: ErrorEventSource | string
  message: string
  stack_trace: string | null
  path: string | null
  method: string | null
  status_code: number | null
  user_id: number | null
  user_agent: string | null
  page_url: string | null
  extra: Record<string, unknown> | null
  created_at: string | null
}

export type AuditLogCategory = 'auth' | 'admin'

export interface AuditLogEntry {
  id: number
  action: string
  category: AuditLogCategory | string
  actor_user_id: number | null
  actor_email: string | null
  resource_type: string | null
  resource_id: string | null
  success: boolean
  ip_address: string | null
  user_agent: string | null
  details: Record<string, unknown> | null
  created_at: string | null
}

export interface SiteBrand {
  brand_name: string
  tagline: string
  logo_url: string
  primary_color: string
  secondary_color: string
  sign_in_url: string
  access_platform_url: string
  page_title: string
}

export interface NavMenuItem {
  id: number
  location: string
  label: string
  href: string
  sort_order: number
  is_external: boolean
}

export interface LandingPayload {
  brand: SiteBrand
  menus: Record<string, NavMenuItem[]>
  landing: Record<string, unknown>
}

export interface BlogPostListItem {
  id: number
  title: string
  slug: string
  excerpt: string
  cover_image_url: string | null
  status: string
  published_at: string | null
  created_at: string
  updated_at: string
}

export interface BlogPost extends BlogPostListItem {
  body_markdown: string
  author_id: number | null
}

export type BugReportStatus = 'open' | 'fixed'

export interface BugReport {
  id: number
  user_id: number | null
  user_email: string | null
  description: string
  image_url: string | null
  page_url: string | null
  status: BugReportStatus | string
  created_at: string
  fixed_at: string | null
}

export interface BugReportSettings {
  enabled: boolean
}

// ---------------------------------------------------------------------------
// AgentAdaptive
// ---------------------------------------------------------------------------

export type AdaptiveJobStatus =
  | 'queued'
  | 'clarifying'
  | 'designing'
  | 'building'
  | 'tuning'
  | 'reporting'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type AdaptiveJobStage =
  | 'queued'
  | 'clarify'
  | 'design'
  | 'build'
  | 'tune'
  | 'report'
  | 'done'
  | 'error'

export interface AdaptiveJobOptions {
  enable_tuning?: boolean
  target_rms_frac?: number
  max_tuning_rounds?: number
  skip_clarify?: boolean
  model?: string | null
  description?: string
  sim_time?: number
  solver_step?: number
  x0?: number[]
  references?: Record<string, string>
  tuning_objectives?: Record<string, number> | null
}

export interface AdaptiveJobCreateRequest {
  system_spec?: Record<string, unknown> | null
  options?: AdaptiveJobOptions
  user_id?: number | null
  project_id?: string | null
}

export interface AdaptiveJobCreateResponse {
  job_id: string
  status: AdaptiveJobStatus
  stage: AdaptiveJobStage
  message: string
}

export interface AdaptiveClarifyRequest {
  answer: string
  force_finish?: boolean
}

export interface AdaptiveClarifyResponse {
  job_id: string
  status: AdaptiveJobStatus
  stage: AdaptiveJobStage
  clarifier_status: 'continue' | 'complete' | 'error' | 'skipped'
  reply: string
  round: number
}

export interface AdaptiveJobProgressEvent {
  kind: string
  stage: string
  text: string
  round?: number | null
  ts?: number | null
  extra?: Record<string, unknown>
}

export interface AdaptiveJobStatusResponse {
  job_id: string
  status: AdaptiveJobStatus
  stage: AdaptiveJobStage
  message: string
  error?: string | null
  round: number
  clarify_pending: boolean
  last_clarifier_reply?: string | null
  progress: AdaptiveJobProgressEvent[]
  created_at: string
  updated_at: string
  user_id?: number | null
  project_id?: string | null
  options?: AdaptiveJobOptions | null
}

export interface AdaptiveDiagnosisOption {
  label?: string
  value?: string | number
  [key: string]: unknown
}

export interface AdaptiveDiagnosisSuggestion {
  title?: string
  field?: string
  lever?: string | null
  target?: string | null
  detail?: string
  rationale?: string
  text?: string
  options?: AdaptiveDiagnosisOption[] | unknown
  values?: unknown
  apply_values?: unknown
  [key: string]: unknown
}

export interface AdaptiveDiagnosisReport {
  headline?: string
  explanation?: string | string[]
  cause?: string
  summary?: string
  diagnosis?: string
  suggestions?: AdaptiveDiagnosisSuggestion[]
  chart_series?: string[]
  error?: string
  [key: string]: unknown
}

export interface AdaptiveDiagnosis {
  report?: AdaptiveDiagnosisReport | null
  evidence?: Record<string, unknown> | null
  chart_data?: Record<string, unknown> | null
  [key: string]: unknown
}

/** Patch applied when user picks a concrete diagnoser suggestion option. */
export interface DiagnosisApplyPatch {
  field?: string
  lever?: string
  value?: unknown
  label?: string
}

export interface AdaptiveDiagnosisChatRequest {
  message: string
  history?: Array<{ role: string; text?: string; content?: string }>
}

export interface AdaptiveDiagnosisChatResponse {
  reply: string
  usage?: Record<string, unknown> | null
}

export type MpcDiagnosis = AdaptiveDiagnosis
export type MPCDiagnosisChatRequest = AdaptiveDiagnosisChatRequest
export type MPCDiagnosisChatResponse = AdaptiveDiagnosisChatResponse

export interface AdaptiveJobResultsResponse {
  job_id: string
  status: AdaptiveJobStatus
  stage: AdaptiveJobStage
  score?: number | null
  success?: boolean | null
  design_grade?: {
    rating: number
    comment?: string | null
    created_at?: string | null
  } | null
  session_metadata?: SessionMetadata | null
  abstract?: string | null
  report?: string | null
  method?: string | null
  control_law?: string | null
  stability_proof?: string | null
  export_script?: string | null
  final_metrics?: Record<string, unknown> | null
  tuning_log?: Array<Record<string, unknown>>
  tuning_best?: Record<string, unknown> | null
  system_spec?: Record<string, unknown> | null
  clarification_record?: Array<Record<string, unknown>>
  usage?: Record<string, unknown> | null
  series?: {
    t?: number[]
    x?: number[][]
    u?: number[][]
    xd?: number[][]
    d_hat?: number[][]
    names?: string[]
    [key: string]: unknown
  } | null
  diagnosis?: AdaptiveDiagnosis | null
  error?: string | null
}

export interface AdaptiveJobSummary {
  job_id: string
  status: AdaptiveJobStatus
  stage: AdaptiveJobStage
  system_name?: string | null
  score?: number | null
  success?: boolean | null
  rating?: number | null
  created_at: string
  updated_at: string
  user_id?: number | null
}

// ---------------------------------------------------------------------------
// AgentMPC
// ---------------------------------------------------------------------------

export type MPCJobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type MPCJobStage =
  | 'queued'
  | 'scenarist'
  | 'actor'
  | 'evaluator'
  | 'terminator'
  | 'critic'
  | 'juror'
  | 'done'
  | 'error'

export interface MPCJobOptions {
  max_iterations?: number
  prediction_horizon?: number
  control_horizon?: number
  dt_mpc?: number
  simulation_time?: number
  ui_scenario_level?: number
  user_guidance?: string
  min_explore_iterations?: number
  exploration_intensity?: number
  use_ui_graph?: boolean
  seed_params?: Record<string, unknown> | null
  model?: string | null
  system_name?: string
  trajectory_mode?: 'reg' | 'sin' | 'pulse' | 'custom'
  trajectory_amplitude?: number
  trajectory_frequency?: number
  trajectory_pulse_start?: number
  trajectory_pulse_end?: number
  noise_std?: number
  q_weights?: number[]
  r_weights?: number[]
  p_weights?: number[]
  cost_weights?: Record<string, number>
  custom_drift_pct?: number
  disturbance_amplitude?: number
  disturbance_start?: number
  disturbance_type?: 'none' | 'step' | 'pulse'
  target_state_indices?: number[]
  trajectory_per_state_modes?: string[]
}

export interface MPCDynamicsInput {
  plugin_path?: string | null
  plugin_id?: string | null
  source?: string | null
}

export interface MPCDiagnosticsRequest {
  dynamics?: MPCDynamicsInput | null
  dt?: number
  sim_time?: number
  u_step_fraction?: number
}

export interface MPCDiagnosticsResponse {
  eigenvalues: Array<{ real: number; imag: number }>
  is_stable: boolean
  is_controllable: boolean
  controllability_rank: number
  n_states: number
  n_inputs: number
  state_names: string[]
  input_names: string[]
  suggested_dt: number
  bryson_q: number[]
  bryson_r: number[]
  probe_trajectory: {
    t?: number[]
    x?: Record<string, number[]>
    ranges?: Record<string, number>
    step_mag?: Record<string, number>
  }
  notes: string[]
  error?: string | null
}

export interface MPCSimulateRequest {
  job_id?: string | null
  dynamics?: MPCDynamicsInput | null
  np?: number
  nc?: number
  dt?: number
  sim_time?: number
  q?: number[]
  r?: number[]
  p?: number[]
  trajectory_mode?: string
  trajectory_amplitude?: number
  trajectory_frequency?: number
  trajectory_pulse_start?: number
  trajectory_pulse_end?: number
  noise_std?: number
  scenario_level?: number
}

export interface MPCSimulateResponse {
  series: {
    t: number[]
    x: Record<string, number[]>
    xd: Record<string, number[]>
    u: Record<string, number[]>
    names: string[]
    input_names: string[]
    bounds?: {
      x_lo?: Array<number | null> | null
      x_hi?: Array<number | null> | null
      u_lo?: Array<number | null> | null
      u_hi?: Array<number | null> | null
    }
    per_state_metrics?: Array<{
      name: string
      mse: number
      overshoot: number
      iae: number
      ise: number
    }>
  }
  metrics: {
    mse: number
    overshoot: number
    settling_time: number
    control_effort: number
    integral_abs_error: number
    integral_sq_error: number
    is_regulation: boolean
    settled: boolean
    per_state_mse?: Record<string, number>
  }
  solve_time_ms: number
  unstable: boolean
  unstable_reason?: string | null
  error?: string | null
}

export interface MPCJobCreateRequest {
  dynamics?: MPCDynamicsInput | null
  options?: MPCJobOptions
  user_id?: number | null
  project_id?: string | null
}

export interface MPCJobCreateResponse {
  job_id: string
  status: MPCJobStatus
  stage: MPCJobStage
  message: string
}

export interface MPCJobProgressEvent {
  kind: string
  stage: string
  text: string
  round?: number | null
  ts?: number | null
  extra?: Record<string, unknown>
}

export interface MPCJobStatusResponse {
  job_id: string
  status: MPCJobStatus
  stage: MPCJobStage
  message: string
  error?: string | null
  iteration: number
  max_iterations: number
  progress: MPCJobProgressEvent[]
  created_at: string
  updated_at: string
  user_id?: number | null
  project_id?: string | null
  options?: MPCJobOptions | null
  system_name?: string | null
  series?: Record<string, unknown> | null
  baseline_series?: Record<string, unknown> | null
  best_params?: Record<string, unknown> | null
  best_mse?: number | null
  mse_history?: Array<number | null>
  overshoot_history?: Array<number | null>
  settling_history?: Array<number | null>
  effort_history?: Array<number | null>
  params_history?: Array<Record<string, unknown>>
  session_metadata?: Record<string, unknown> | null
  usage?: Record<string, unknown> | null
}

export interface MPCJobResultsResponse {
  job_id: string
  status: MPCJobStatus
  stage: MPCJobStage
  score?: number | null
  success?: boolean | null
  design_grade?: {
    rating: number
    comment?: string | null
    created_at?: string | null
  } | null
  session_metadata?: SessionMetadata | null
  best_params?: Record<string, unknown> | null
  best_mse?: number | null
  iteration: number
  termination_reason?: string | null
  mse_history: Array<number | null>
  overshoot_history: Array<number | null>
  settling_history: Array<number | null>
  effort_history: Array<number | null>
  params_history: Array<Record<string, unknown>>
  history: Array<Record<string, unknown> | string>
  report?: string | null
  export_script?: string | null
  metrics?: Record<string, unknown> | null
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
    total_cost?: number
    model?: string
    [key: string]: unknown
  } | null
  diagnostics?: Record<string, unknown> | null
  diagnosis?: MpcDiagnosis | Record<string, unknown> | null
  series?: {
    t: number[]
    x: Record<string, number[]>
    xd: Record<string, number[]>
    u: Record<string, number[]>
    names: string[]
    input_names: string[]
    bounds?: {
      x_lo?: Array<number | null> | null
      x_hi?: Array<number | null> | null
      u_lo?: Array<number | null> | null
      u_hi?: Array<number | null> | null
    }
    per_state_metrics?: Array<{
      name: string
      mse: number
      overshoot: number
      iae: number
      ise: number
    }>
    [key: string]: unknown
  } | null
  baseline_series?: {
    t: number[]
    x: Record<string, number[]>
    xd: Record<string, number[]>
    u: Record<string, number[]>
    names: string[]
    input_names: string[]
    bounds?: {
      x_lo?: Array<number | null> | null
      x_hi?: Array<number | null> | null
      u_lo?: Array<number | null> | null
      u_hi?: Array<number | null> | null
    }
    per_state_metrics?: Array<{
      name: string
      mse: number
      overshoot: number
      iae: number
      ise: number
    }>
    [key: string]: unknown
  } | null
  error?: string | null
}

export interface MPCJobSummary {
  job_id: string
  status: MPCJobStatus
  stage: MPCJobStage
  system_name?: string | null
  score?: number | null
  success?: boolean | null
  rating?: number | null
  created_at: string
  updated_at: string
  user_id?: number | null
}

// ---------------------------------------------------------------------------
// Pre-Launch & Plant Artifacts
// ---------------------------------------------------------------------------

export interface PreLaunchConfig {
  total_simulation_time: number
  solver_sample_time: number
  initial_state: number[]
  default_target: number[]
  trajectory_mode?: 'reg' | 'sin' | 'pulse'
  trajectory_amplitude?: number
  trajectory_frequency?: number
  trajectory_offset?: number
}

export interface PlantPayload {
  system_name: string
  python_code: string
  metadata?: Record<string, unknown> | null
}

export interface ArtifactCreateRequest {
  pre_launch: PreLaunchConfig
  plant?: PlantPayload | null
  conversation_id?: number | null
}

export interface ArtifactSummary {
  artifact_id: string
  system_name: string
  created_at: string
  version: string
}

export interface ArtifactCreateResponse {
  artifact_id: string
  system_name: string
  created_at: string
  version: string
  warnings: string[]
}

export interface ArtifactDetail {
  artifact_id: string
  system_name: string
  created_at: string
  version: string
  plant: Record<string, unknown>
  pre_launch: Record<string, unknown>
  module_specific: Record<string, unknown>
}

export interface ArtifactPluginResponse {
  artifact_id: string
  plugin_path: string
  source: string
}

export interface ValidationRequest {
  plant?: PlantPayload | null
  pre_launch?: PreLaunchConfig | null
  conversation_id?: number | null
}

export interface ValidationResponse {
  ok: boolean
  errors: string[]
  warnings: string[]
}

export interface CreditRates {
  per_1k_tokens: number | string
  per_minute: number | string
  min_job_charge: number | string
}

export interface CreditDashboard {
  bonus_balance: number | string
  daily_balance: number | string
  spendable: number | string
  daily_allotment: number | string
  daily_date: string | null
  daily_reset_at: string
  today_spent: number | string
  used_up: boolean
  used_up_message: string | null
  referral_code: string | null
  referral_link: string | null
  hard_gate_enabled: boolean
  rates: CreditRates
}

export interface CreditLedgerEntry {
  id: number
  amount: number | string
  balance_after: number | string
  entry_type: string
  usage_session_id: number | null
  note: string
  created_at: string
}

export interface CreditUsageSession {
  id: number
  module: string
  job_id: string | null
  status: string
  started_at: string
  ended_at: string | null
  duration_seconds: number
  prompt_tokens: number
  completion_tokens: number
  credits_charged: number | string
  live_credits_estimate: number | string
}

export interface CreditSettings {
  new_user_bonus: number | string
  referral_inviter_bonus: number | string
  referral_invitee_bonus: number | string
  daily_allotment: number | string
  per_1k_tokens: number | string
  per_minute: number | string
  min_job_charge: number | string
  hard_gate_enabled: boolean
}

export interface CreditSettingsUpdate {
  new_user_bonus?: number
  referral_inviter_bonus?: number
  referral_invitee_bonus?: number
  daily_allotment?: number
  per_1k_tokens?: number
  per_minute?: number
  min_job_charge?: number
  hard_gate_enabled?: boolean
}

export interface AdminUserCredits {
  user_id: number
  bonus_balance: number | string
  daily_balance: number | string
  spendable: number | string
  daily_date: string | null
  daily_reset_at: string | null
  used_up: boolean
  used_up_message: string | null
  referral_code: string | null
  referred_by_user_id: number | null
  new_user_bonus_granted_at: string | null
  ledger: CreditLedgerEntry[]
  sessions: CreditUsageSession[]
}


