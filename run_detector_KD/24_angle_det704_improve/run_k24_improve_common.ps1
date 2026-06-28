param(
    [ValidateSet(
        "run1_1stage_nodc_rmax005",
        "run2_1stage_weakphys_rmax005",
        "run3_1stage_tinyphys_rmax003",
        "run4_1stage_weak_stage_dc",
        "run5_2stage_tied_small_nodc"
    )]
    [string]$Experiment,
    [ValidateSet("train", "test")]
    [string]$Mode = "train",
    [double]$NoiseLevel = 0.1,
    [string]$OutputTag = "",
    [string]$OfflineDataDir = "",
    [string]$OfflineTrainDataset = "",
    [string]$OfflineValDataset = "",
    [string]$OfflineEvalDataset = "",
    [switch]$UseSheppLoganTestAsVal,
    [ValidateSet("best", "final")]
    [string]$ModelChoice = "best",
    [string]$ModelPath = "",
    [int]$NumSamples = 500,
    [int]$OfflineBatchSize = 20,
    [string]$ResultPrefix = "",
    [string]$ResultDir = "",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$PythonBin = "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe",
    [switch]$AllowOnlineFallback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Set-RunEnv {
    param(
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [AllowEmptyString()] [string]$Value
    )
    Set-Item -Path "Env:$Name" -Value $Value
}

function Require-File {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$Message
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Message Missing file: $Path"
    }
}

function Resolve-InputPath {
    param([Parameter(Mandatory = $true)] [AllowEmptyString()] [string]$Path)
    $trimmed = $Path.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) { return "" }
    if ([System.IO.Path]::IsPathRooted($trimmed)) { return $trimmed }
    return (Join-Path $ProjectRoot $trimmed)
}

function Resolve-DefaultOfflineDataset {
    param(
        [Parameter(Mandatory = $true)] [string]$Directory,
        [Parameter(Mandatory = $true)] [string]$ExactName,
        [Parameter(Mandatory = $true)] [string]$Pattern
    )
    if ([string]::IsNullOrWhiteSpace($Directory)) { return "" }
    $exactPath = Join-Path $Directory $ExactName
    if (Test-Path -LiteralPath $exactPath -PathType Leaf) {
        return (Resolve-Path -LiteralPath $exactPath).Path
    }
    if (Test-Path -LiteralPath $Directory -PathType Container) {
        $match = Get-ChildItem -LiteralPath $Directory -Filter $Pattern -File |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -ne $match) { return $match.FullName }
    }
    return ""
}

$experimentConfigs = @{
    "run1_1stage_nodc_rmax005" = @{
        Description = "one-stage residual, no explicit update, no stage DC, residual max 0.05"
        RefinerStages = "1"; ShareWeights = "1"; StageDcEnabled = "0"; StageDcCgIters = "4"; StageDcDamping = "1e-2"
        ExplicitEnabled = "0"; ExplicitAlphaInit = "0.0"; ExplicitMax = "0.0"; PhysicsGateMode = "spatial"
        ResidualMax = "0.05"; IntermediateEnabled = "0"; IntermediateStart = "0.0"; IntermediateEnd = "0.0"
        BaseLr = "1e-4"; LrSchedule = "constant_cosine"; LrConstantSteps = "4500"; LrMinFactor = "0.2"; LrWarmup = "300"; ScalarLrRatio = "0.1"
        BaseChannels = "64"; Depth = "4"
    }
    "run2_1stage_weakphys_rmax005" = @{
        Description = "one-stage residual, weak explicit physics update, no stage DC, residual max 0.05"
        RefinerStages = "1"; ShareWeights = "1"; StageDcEnabled = "0"; StageDcCgIters = "4"; StageDcDamping = "1e-2"
        ExplicitEnabled = "1"; ExplicitAlphaInit = "0.02"; ExplicitMax = "0.05"; PhysicsGateMode = "spatial"
        ResidualMax = "0.05"; IntermediateEnabled = "0"; IntermediateStart = "0.0"; IntermediateEnd = "0.0"
        BaseLr = "1e-4"; LrSchedule = "constant_cosine"; LrConstantSteps = "4500"; LrMinFactor = "0.2"; LrWarmup = "300"; ScalarLrRatio = "0.1"
        BaseChannels = "64"; Depth = "4"
    }
    "run3_1stage_tinyphys_rmax003" = @{
        Description = "one-stage residual, tiny explicit physics update, no stage DC, residual max 0.03"
        RefinerStages = "1"; ShareWeights = "1"; StageDcEnabled = "0"; StageDcCgIters = "4"; StageDcDamping = "1e-2"
        ExplicitEnabled = "1"; ExplicitAlphaInit = "0.01"; ExplicitMax = "0.03"; PhysicsGateMode = "spatial"
        ResidualMax = "0.03"; IntermediateEnabled = "0"; IntermediateStart = "0.0"; IntermediateEnd = "0.0"
        BaseLr = "1e-4"; LrSchedule = "constant_cosine"; LrConstantSteps = "4500"; LrMinFactor = "0.2"; LrWarmup = "300"; ScalarLrRatio = "0.1"
        BaseChannels = "64"; Depth = "4"
    }
    "run4_1stage_weak_stage_dc" = @{
        Description = "one-stage residual, weak stage DC, no explicit update, residual max 0.05"
        RefinerStages = "1"; ShareWeights = "1"; StageDcEnabled = "1"; StageDcCgIters = "2"; StageDcDamping = "1e-1"
        ExplicitEnabled = "0"; ExplicitAlphaInit = "0.0"; ExplicitMax = "0.0"; PhysicsGateMode = "spatial"
        ResidualMax = "0.05"; IntermediateEnabled = "0"; IntermediateStart = "0.0"; IntermediateEnd = "0.0"
        BaseLr = "1e-4"; LrSchedule = "constant_cosine"; LrConstantSteps = "4500"; LrMinFactor = "0.2"; LrWarmup = "300"; ScalarLrRatio = "0.1"
        BaseChannels = "64"; Depth = "4"
    }
    "run5_2stage_tied_small_nodc" = @{
        Description = "two-stage shared residual, tiny explicit physics update, no stage DC, residual max 0.03"
        RefinerStages = "2"; ShareWeights = "1"; StageDcEnabled = "0"; StageDcCgIters = "4"; StageDcDamping = "1e-2"
        ExplicitEnabled = "1"; ExplicitAlphaInit = "0.01"; ExplicitMax = "0.03"; PhysicsGateMode = "spatial"
        ResidualMax = "0.03"; IntermediateEnabled = "1"; IntermediateStart = "0.2"; IntermediateEnd = "0.5"
        BaseLr = "1e-4"; LrSchedule = "constant_cosine"; LrConstantSteps = "4500"; LrMinFactor = "0.2"; LrWarmup = "300"; ScalarLrRatio = "0.1"
        BaseChannels = "64"; Depth = "4"
    }
}

if (-not $experimentConfigs.ContainsKey($Experiment)) { throw "Unknown experiment: $Experiment" }
$cfg = $experimentConfigs[$Experiment]

$K = 24
$DetectorSamples = 704
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
Set-Location -LiteralPath $ProjectRoot
if ($PythonBin -match "[\\/]" -and -not (Test-Path -LiteralPath $PythonBin -PathType Leaf)) {
    throw "Python executable not found: $PythonBin"
}

$alphaJson = Join-Path $ProjectRoot "data\alpha24_tv\alpha_selected24_dopt_hard_gap_6_9.json"
Require-File -Path $alphaJson -Message "Selected-angle JSON not found."

$cnnAngleIndices = (0..($K - 1)) -join ","
$noiseText = $NoiseLevel.ToString("G", [System.Globalization.CultureInfo]::InvariantCulture)
$offlineNoiseTag = $noiseText.Replace(".", "p").Replace("-", "m")
$noiseTag = "n" + $offlineNoiseTag
$outputTag = $OutputTag.Trim()
if ([string]::IsNullOrWhiteSpace($outputTag)) {
    $outputTag = "a24_d704_${noiseTag}_admm80_$Experiment"
}

$experimentRoot = Join-Path $PSScriptRoot $Experiment
$logDir = Join-Path $experimentRoot "log"
$defaultResultDir = Join-Path $experimentRoot "result"
$checkpointDir = Join-Path $experimentRoot "checkpoints"
New-Item -ItemType Directory -Force -Path $logDir, $defaultResultDir, $checkpointDir | Out-Null

$kdRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$resolvedOfflineDataDir = Resolve-InputPath -Path $OfflineDataDir
if ([string]::IsNullOrWhiteSpace($resolvedOfflineDataDir)) {
    $resolvedOfflineDataDir = Join-Path $kdRoot "24_angle_det704\offline_data"
}

$fileSuffix = "alpha24_noise${offlineNoiseTag}_det704_edgewsubset"
$offlineTrain = Resolve-InputPath -Path $OfflineTrainDataset
if ([string]::IsNullOrWhiteSpace($offlineTrain)) {
    $offlineTrain = Resolve-DefaultOfflineDataset -Directory $resolvedOfflineDataDir -ExactName ("train8000_random_ellipses_tvinit_{0}.pt" -f $fileSuffix) -Pattern ("train*_random_ellipses_tvinit_{0}.pt" -f $fileSuffix)
}
$offlineVal = Resolve-InputPath -Path $OfflineValDataset
$valDataSource = "random_ellipses"
if ($UseSheppLoganTestAsVal -and [string]::IsNullOrWhiteSpace($offlineVal)) {
    $offlineVal = Resolve-DefaultOfflineDataset -Directory $resolvedOfflineDataDir -ExactName ("test500_shepp_logan_tvinit_{0}.pt" -f $fileSuffix) -Pattern ("test*_shepp_logan_tvinit_{0}.pt" -f $fileSuffix)
    $valDataSource = "shepp_logan"
} elseif ([string]::IsNullOrWhiteSpace($offlineVal)) {
    $offlineVal = Resolve-DefaultOfflineDataset -Directory $resolvedOfflineDataDir -ExactName ("val500_random_ellipses_tvinit_{0}.pt" -f $fileSuffix) -Pattern ("val*_random_ellipses_tvinit_{0}.pt" -f $fileSuffix)
}
if (-not [string]::IsNullOrWhiteSpace($offlineVal)) {
    $offlineValName = [System.IO.Path]::GetFileName($offlineVal).ToLowerInvariant()
    if ($offlineValName.Contains("shepp_logan")) { $valDataSource = "shepp_logan" }
}
$offlineEval = Resolve-InputPath -Path $OfflineEvalDataset
if ([string]::IsNullOrWhiteSpace($offlineEval)) {
    $offlineEval = Resolve-DefaultOfflineDataset -Directory $resolvedOfflineDataDir -ExactName ("test500_shepp_logan_tvinit_{0}.pt" -f $fileSuffix) -Pattern ("test*_shepp_logan_tvinit_{0}.pt" -f $fileSuffix)
}

if ($Mode -eq "train" -and -not $DryRun -and -not $AllowOnlineFallback) {
    if ([string]::IsNullOrWhiteSpace($offlineTrain) -or -not (Test-Path -LiteralPath $offlineTrain -PathType Leaf)) { throw "Offline train dataset not found. Pass -OfflineDataDir/-OfflineTrainDataset or -AllowOnlineFallback." }
    if ([string]::IsNullOrWhiteSpace($offlineVal) -or -not (Test-Path -LiteralPath $offlineVal -PathType Leaf)) { throw "Offline validation dataset not found. Pass -OfflineDataDir/-OfflineValDataset or -AllowOnlineFallback." }
}
if ($AllowOnlineFallback -and -not (Test-Path -LiteralPath $offlineTrain -PathType Leaf)) { $offlineTrain = "" }
if ($AllowOnlineFallback -and -not (Test-Path -LiteralPath $offlineVal -PathType Leaf)) { $offlineVal = "" }

$resolvedResultDir = Resolve-InputPath -Path $ResultDir
if ([string]::IsNullOrWhiteSpace($resolvedResultDir)) { $resolvedResultDir = $defaultResultDir }
$resolvedResultPrefix = $ResultPrefix.Trim()
if ([string]::IsNullOrWhiteSpace($resolvedResultPrefix)) { $resolvedResultPrefix = $outputTag }

$modelFileKind = if ($ModelChoice -eq "best") { "best_model" } else { "model" }
$defaultModelPath = Join-Path $checkpointDir ("theoretical_ct_{0}_{1}.pth" -f $outputTag, $modelFileKind)
$compactModelPath = $defaultModelPath -replace "\.pth$", "_compact.pth"
$resolvedModelPath = Resolve-InputPath -Path $ModelPath
if ([string]::IsNullOrWhiteSpace($resolvedModelPath)) {
    $resolvedModelPath = if (Test-Path -LiteralPath $compactModelPath -PathType Leaf) { $compactModelPath } else { $defaultModelPath }
}

Set-RunEnv "PROJECT_ROOT" $ProjectRoot
Set-RunEnv "PYTHONUTF8" "1"
Set-RunEnv "PYTHON_BIN" $PythonBin
Set-RunEnv "EXPERIMENT_PROFILE_OVERRIDE" "alpha_condition"
Set-RunEnv "ALPHA_CONDITION_TOP_K_OVERRIDE" "24"
Set-RunEnv "ALPHA_CONDITION_JSON_OVERRIDE" $alphaJson
Set-RunEnv "ALPHA_GRAM_CACHE_DIR_OVERRIDE" (Join-Path $ProjectRoot "data\alpha_gram_cache")
Set-RunEnv "NUM_ANGLES_TOTAL_OVERRIDE" "24"
Set-RunEnv "CNN_ANGLE_INDICES_OVERRIDE" $cnnAngleIndices
Set-RunEnv "CNN_NUM_ANGLES_OVERRIDE" "24"
Set-RunEnv "MULTI_ANGLE_SOLVER_MODE_OVERRIDE" "stacked_tikhonov"
Set-RunEnv "THEORETICAL_FORMULA_MODE_OVERRIDE" "alpha_continuous"
Set-RunEnv "SAMPLING_MODE_OVERRIDE" "shifted_lattice_edge_weighted_subset"
Set-RunEnv "NUM_DETECTOR_SAMPLES_OVERRIDE" "704"
Set-RunEnv "DETECTOR_PHASE_OVERRIDE" "0.5"
Set-RunEnv "DETECTOR_MARGIN_RATIO_OVERRIDE" "0.0"
Set-RunEnv "INIT_METHOD_OVERRIDE" "l2_tv_admm"
Set-RunEnv "LAMBDA_SELECT_MODE_OVERRIDE" "morozov"
Set-RunEnv "MOROZOV_FORM_OVERRIDE" "constrained"
Set-RunEnv "MOROZOV_NOISE_RADIUS_MODE_OVERRIDE" "rms"
Set-RunEnv "MOROZOV_TAU_OVERRIDE" "1.0"
Set-RunEnv "L1_INIT_ADMM_ITERS_OVERRIDE" "80"
Set-RunEnv "L1_INIT_ADMM_CG_ITERS_OVERRIDE" "30"
Set-RunEnv "L1_INIT_ADMM_CG_TOL_OVERRIDE" "1e-4"
Set-RunEnv "L1_INIT_ADMM_RHO_DATA_OVERRIDE" "1.0"
Set-RunEnv "L1_INIT_ADMM_RHO_REG_OVERRIDE" "1.0"
Set-RunEnv "REGULARIZER_TYPE_OVERRIDE" "dirichlet"
Set-RunEnv "NOISE_MODE_OVERRIDE" "multiplicative"
Set-RunEnv "NOISE_LEVEL_OVERRIDE" $noiseText
Set-RunEnv "MODEL_ARCH_OVERRIDE" "tv_pc_cascade_unet"
Set-RunEnv "REFINER_INPUT_MODE_OVERRIDE" "u2_stacked"
Set-RunEnv "DATA_FIDELITY_CHANNEL_MODE_OVERRIDE" "stacked_selected"
Set-RunEnv "UNET_BACKBONE_OVERRIDE" "rad_unet"
Set-RunEnv "UNET_BASE_CHANNELS_OVERRIDE" ([string]$cfg.BaseChannels)
Set-RunEnv "UNET_DEPTH_OVERRIDE" ([string]$cfg.Depth)
Set-RunEnv "UNET_RESIDUAL_MAX_OVERRIDE" ([string]$cfg.ResidualMax)
Set-RunEnv "PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE" "1"
Set-RunEnv "PHYSICS_RESIDUAL_MODE_OVERRIDE" "stacked_selected_cg"
Set-RunEnv "PHYSICS_RESIDUAL_DAMPING_OVERRIDE" "1e-2"
Set-RunEnv "PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE" "8"
Set-RunEnv "PHYSICS_RESIDUAL_DETACH_OVERRIDE" "1"
Set-RunEnv "PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE" "1"
Set-RunEnv "PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE" ([string]$cfg.ExplicitEnabled)
Set-RunEnv "PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE" ([string]$cfg.ExplicitAlphaInit)
Set-RunEnv "PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE" ([string]$cfg.ExplicitMax)
Set-RunEnv "PHYSICS_GATE_MODE_OVERRIDE" ([string]$cfg.PhysicsGateMode)
Set-RunEnv "REFINER_STAGES_OVERRIDE" ([string]$cfg.RefinerStages)
Set-RunEnv "REFINER_SHARE_WEIGHTS_OVERRIDE" ([string]$cfg.ShareWeights)
Set-RunEnv "REFINER_STAGE_DC_ENABLED_OVERRIDE" ([string]$cfg.StageDcEnabled)
Set-RunEnv "REFINER_STAGE_DC_CG_ITERS_OVERRIDE" ([string]$cfg.StageDcCgIters)
Set-RunEnv "REFINER_STAGE_DC_DAMPING_OVERRIDE" ([string]$cfg.StageDcDamping)
Set-RunEnv "REFINER_STAGE_DC_DETACH_OVERRIDE" "1"
Set-RunEnv "REFINER_STAGE_DC_NORMALIZE_OVERRIDE" "1"
Set-RunEnv "BASE_LR_OVERRIDE" ([string]$cfg.BaseLr)
Set-RunEnv "LR_SCHEDULE_OVERRIDE" ([string]$cfg.LrSchedule)
Set-RunEnv "LR_CONSTANT_STEPS_OVERRIDE" ([string]$cfg.LrConstantSteps)
Set-RunEnv "LR_MIN_FACTOR_OVERRIDE" ([string]$cfg.LrMinFactor)
Set-RunEnv "LR_WARMUP_STEPS_OVERRIDE" ([string]$cfg.LrWarmup)
Set-RunEnv "SCALAR_LR_RATIO_OVERRIDE" ([string]$cfg.ScalarLrRatio)
Set-RunEnv "INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE" ([string]$cfg.IntermediateEnabled)
Set-RunEnv "INTERMEDIATE_SUPERVISION_WEIGHT_START_OVERRIDE" ([string]$cfg.IntermediateStart)
Set-RunEnv "INTERMEDIATE_SUPERVISION_WEIGHT_END_OVERRIDE" ([string]$cfg.IntermediateEnd)
Set-RunEnv "DETACH_PHYSICAL_GRADS_OVERRIDE" "0"
Set-RunEnv "N_DATA_OVERRIDE" "4"
Set-RunEnv "N_TRAIN_OVERRIDE" "8000"
Set-RunEnv "VALIDATION_INTERVAL_OVERRIDE" "50"
Set-RunEnv "VAL_RANDOM_SUBSAMPLE_OVERRIDE" "1"
Set-RunEnv "VAL_SUBSAMPLE_SIZE_OVERRIDE" "500"
Set-RunEnv "VAL_BATCH_SIZE_OVERRIDE" "100"
Set-RunEnv "COMPACT_CHECKPOINTS_OVERRIDE" "1"
Set-RunEnv "OUTPUT_TAG_OVERRIDE" $outputTag
Set-RunEnv "LOG_DIR_OVERRIDE" $logDir
Set-RunEnv "RESULTS_DIR_OVERRIDE" $resolvedResultDir
Set-RunEnv "RESULT_DIR_OVERRIDE" $resolvedResultDir
Set-RunEnv "MODEL_DIR_OVERRIDE" $checkpointDir
Set-RunEnv "CHECKPOINT_DIR_OVERRIDE" $checkpointDir
Set-RunEnv "MODEL_PATH_OVERRIDE" (Join-Path $checkpointDir ("theoretical_ct_{0}_model.pth" -f $outputTag))
Set-RunEnv "BEST_MODEL_PATH_OVERRIDE" (Join-Path $checkpointDir ("theoretical_ct_{0}_best_model.pth" -f $outputTag))
Set-RunEnv "MODEL_LOAD_PATH_OVERRIDE" $resolvedModelPath

Write-Host "[$Mode] experiment=$Experiment"
Write-Host "[$Mode] description=$($cfg.Description)"
Write-Host "[$Mode] project=$ProjectRoot"
Write-Host "[$Mode] python=$PythonBin"
Write-Host "[$Mode] K=24 detector_samples=704 measurements=$($K * $DetectorSamples)"
Write-Host "[$Mode] alpha_json=$alphaJson"
Write-Host "[$Mode] offline_data_dir=$resolvedOfflineDataDir"
Write-Host "[$Mode] init=l2_tv_admm morozov=constrained radius=rms admm=80 cg=30 tol=1e-4"
Write-Host "[$Mode] model=tv_pc_cascade_unet input=u2_stacked backbone=rad_unet stages=$($cfg.RefinerStages) residual_max=$($cfg.ResidualMax)"
Write-Host "[$Mode] explicit_update=$($cfg.ExplicitEnabled) alpha_init=$($cfg.ExplicitAlphaInit) max=$($cfg.ExplicitMax) gate=$($cfg.PhysicsGateMode)"
Write-Host "[$Mode] stage_dc=$($cfg.StageDcEnabled) dc_cg=$($cfg.StageDcCgIters) dc_damping=$($cfg.StageDcDamping)"
Write-Host "[$Mode] lr=$($cfg.BaseLr) schedule=$($cfg.LrSchedule) constant_steps=$($cfg.LrConstantSteps) min_factor=$($cfg.LrMinFactor) warmup=$($cfg.LrWarmup)"
Write-Host "[$Mode] output_tag=$outputTag"
Write-Host "[$Mode] log_dir=$logDir"
Write-Host "[$Mode] result_dir=$resolvedResultDir"
Write-Host "[$Mode] checkpoint_dir=$checkpointDir"

if ($Mode -eq "train") {
    Set-RunEnv "OFFLINE_TRAIN_DATASET_OVERRIDE" $offlineTrain
    Set-RunEnv "OFFLINE_VAL_DATASET_OVERRIDE" $offlineVal
    Set-RunEnv "TRAIN_DATA_SOURCE_OVERRIDE" "random_ellipses"
    Set-RunEnv "VAL_DATA_SOURCE_OVERRIDE" $valDataSource
    $onlineData = if ([string]::IsNullOrWhiteSpace($offlineTrain) -and [string]::IsNullOrWhiteSpace($offlineVal)) { "1" } else { "0" }
    Write-Host "[train] online_data=$onlineData offline_train='$offlineTrain' offline_val='$offlineVal' val_data_source=$valDataSource"
    if ($DryRun) {
        Write-Host "[train] DryRun enabled; environment prepared but training was not started."
        exit 0
    }
    & $PythonBin (Join-Path $ProjectRoot "models\deep_learn\train.py")
    exit $LASTEXITCODE
}

Write-Host "[test] model_choice=$ModelChoice model=$resolvedModelPath"
Write-Host "[test] offline_eval='$offlineEval' num_samples=$NumSamples batch_size=$OfflineBatchSize result_prefix=$resolvedResultPrefix"
if ($DryRun) {
    $modelExistsText = if (Test-Path -LiteralPath $resolvedModelPath -PathType Leaf) { "yes" } else { "no" }
    $evalExistsText = if (-not [string]::IsNullOrWhiteSpace($offlineEval) -and (Test-Path -LiteralPath $offlineEval -PathType Leaf)) { "yes" } else { "no" }
    Write-Host "[test] DryRun enabled; model_exists=$modelExistsText offline_eval_exists=$evalExistsText and evaluation was not started."
    exit 0
}

Require-File -Path $resolvedModelPath -Message "Model checkpoint not found. Use -ModelPath, -OutputTag, or -ModelChoice."
Require-File -Path $offlineEval -Message "Offline eval dataset not found. Use -OfflineDataDir or -OfflineEvalDataset."
& $PythonBin (Join-Path $ProjectRoot "models\deep_learn\evaluate_best_model_offline_val.py") `
    --model-path $resolvedModelPath `
    --offline-val $offlineEval `
    --num-samples $NumSamples `
    --batch-size $OfflineBatchSize `
    --result-dir $resolvedResultDir `
    --result-prefix $resolvedResultPrefix
exit $LASTEXITCODE
