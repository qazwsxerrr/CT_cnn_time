param(
    [ValidateSet("8", "16", "24")]
    [int]$K = 24,
    [int]$DetectorSamples = 0,
    [double]$NoiseLevel = 0.1,
    [string]$OutputTag = "",
    [string]$OfflineTrainDataset = "",
    [string]$OfflineValDataset = "",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$PythonBin = "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe",
    [switch]$AllowOnlineFallback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Set-RunEnv {
    param([Parameter(Mandatory = $true)] [string]$Name, [Parameter(Mandatory = $true)] [AllowEmptyString()] [string]$Value)
    Set-Item -Path "Env:$Name" -Value $Value
}

function Resolve-KDDetectorSamples {
    param([Parameter(Mandatory = $true)] [int]$AngleCount, [int]$Requested)
    if ($Requested -gt 0) { return $Requested }
    if ($AngleCount -eq 8) { return 2048 }
    if ($AngleCount -eq 16) { return 1024 }
    if ($AngleCount -eq 24) { return 704 }
    throw "Unsupported K=$AngleCount."
}

function Resolve-DefaultOfflineDataset {
    param([Parameter(Mandatory = $true)] [string]$Directory, [Parameter(Mandatory = $true)] [string]$ExactName, [Parameter(Mandatory = $true)] [string]$Pattern)
    $exactPath = Join-Path $Directory $ExactName
    if (Test-Path -LiteralPath $exactPath -PathType Leaf) {
        return (Resolve-Path -LiteralPath $exactPath).Path
    }
    if (Test-Path -LiteralPath $Directory -PathType Container) {
        $match = Get-ChildItem -LiteralPath $Directory -Filter $Pattern -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -ne $match) { return $match.FullName }
    }
    return ""
}

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
Set-Location -LiteralPath $ProjectRoot

$DetectorSamples = Resolve-KDDetectorSamples -AngleCount $K -Requested $DetectorSamples
if ($DetectorSamples -le 0) { throw "DetectorSamples must be positive." }
if ($PythonBin -match "[\\/]" -and -not (Test-Path -LiteralPath $PythonBin -PathType Leaf)) { throw "Python executable not found: $PythonBin" }

$angleJsonByK = @{
    8 = "data\alpha8_tv\alpha_selected8_dopt_hard_gap_16_30.json"
    16 = "data\alpha16_tv\alpha_selected16_dopt_hard_gap_9_14.json"
    24 = "data\alpha24_tv\alpha_selected24_dopt_hard_gap_6_9.json"
}
$alphaJson = Join-Path $ProjectRoot $angleJsonByK[$K]
if (-not (Test-Path -LiteralPath $alphaJson -PathType Leaf)) { throw "Selected-angle JSON not found: $alphaJson" }

$cnnAngleIndices = (0..($K - 1)) -join ","
$noiseText = $NoiseLevel.ToString("G", [System.Globalization.CultureInfo]::InvariantCulture)
$offlineNoiseTag = $noiseText.Replace(".", "p").Replace("-", "m")
$noiseTag = "n" + $offlineNoiseTag
$outputTag = $OutputTag.Trim()
if ([string]::IsNullOrWhiteSpace($outputTag)) { $outputTag = "a${K}_d${DetectorSamples}_${noiseTag}_offline" }

$caseDir = "{0}_angle_det{1}" -f $K, $DetectorSamples
$caseRoot = Join-Path $PSScriptRoot $caseDir
$logDir = Join-Path $caseRoot "log"
$resultDir = Join-Path $caseRoot "result"
$checkpointDir = Join-Path $caseRoot "checkpoints"
$offlineDataDir = Join-Path $caseRoot "offline_data"
New-Item -ItemType Directory -Force -Path $logDir, $resultDir, $checkpointDir | Out-Null

$fileSuffix = "alpha${K}_noise${offlineNoiseTag}_det${DetectorSamples}_edgewsubset"
$offlineTrain = $OfflineTrainDataset.Trim()
if ([string]::IsNullOrWhiteSpace($offlineTrain)) {
    $offlineTrain = Resolve-DefaultOfflineDataset -Directory $offlineDataDir -ExactName ("train8000_random_ellipses_tvinit_{0}.pt" -f $fileSuffix) -Pattern ("train*_random_ellipses_tvinit_{0}.pt" -f $fileSuffix)
} elseif (-not [System.IO.Path]::IsPathRooted($offlineTrain)) {
    $offlineTrain = Join-Path $ProjectRoot $offlineTrain
}
$offlineVal = $OfflineValDataset.Trim()
if ([string]::IsNullOrWhiteSpace($offlineVal)) {
    $offlineVal = Resolve-DefaultOfflineDataset -Directory $offlineDataDir -ExactName ("val500_random_ellipses_tvinit_{0}.pt" -f $fileSuffix) -Pattern ("val*_random_ellipses_tvinit_{0}.pt" -f $fileSuffix)
} elseif (-not [System.IO.Path]::IsPathRooted($offlineVal)) {
    $offlineVal = Join-Path $ProjectRoot $offlineVal
}

if (-not $DryRun -and -not $AllowOnlineFallback) {
    if ([string]::IsNullOrWhiteSpace($offlineTrain) -or -not (Test-Path -LiteralPath $offlineTrain -PathType Leaf)) { throw "Offline train dataset not found. Generate offline data first or pass -AllowOnlineFallback." }
    if ([string]::IsNullOrWhiteSpace($offlineVal) -or -not (Test-Path -LiteralPath $offlineVal -PathType Leaf)) { throw "Offline validation dataset not found. Generate offline data first or pass -AllowOnlineFallback." }
}

if ($AllowOnlineFallback -and -not (Test-Path -LiteralPath $offlineTrain -PathType Leaf)) { $offlineTrain = "" }
if ($AllowOnlineFallback -and -not (Test-Path -LiteralPath $offlineVal -PathType Leaf)) { $offlineVal = "" }
$onlineData = if ([string]::IsNullOrWhiteSpace($offlineTrain) -and [string]::IsNullOrWhiteSpace($offlineVal)) { "1" } else { "0" }

Set-RunEnv "OFFLINE_TRAIN_DATASET_OVERRIDE" $offlineTrain
Set-RunEnv "OFFLINE_VAL_DATASET_OVERRIDE" $offlineVal
Set-RunEnv "PROJECT_ROOT" $ProjectRoot
Set-RunEnv "PYTHONUTF8" "1"
Set-RunEnv "PYTHON_BIN" $PythonBin
Set-RunEnv "EXPERIMENT_PROFILE_OVERRIDE" "alpha_condition"
Set-RunEnv "ALPHA_CONDITION_TOP_K_OVERRIDE" ([string]$K)
Set-RunEnv "ALPHA_CONDITION_JSON_OVERRIDE" $alphaJson
Set-RunEnv "ALPHA_GRAM_CACHE_DIR_OVERRIDE" (Join-Path $ProjectRoot "data\alpha_gram_cache")
Set-RunEnv "NUM_ANGLES_TOTAL_OVERRIDE" ([string]$K)
Set-RunEnv "CNN_ANGLE_INDICES_OVERRIDE" $cnnAngleIndices
Set-RunEnv "CNN_NUM_ANGLES_OVERRIDE" ([string]$K)
Set-RunEnv "MULTI_ANGLE_SOLVER_MODE_OVERRIDE" "stacked_tikhonov"
Set-RunEnv "THEORETICAL_FORMULA_MODE_OVERRIDE" "alpha_continuous"
Set-RunEnv "SAMPLING_MODE_OVERRIDE" "shifted_lattice_edge_weighted_subset"
Set-RunEnv "NUM_DETECTOR_SAMPLES_OVERRIDE" ([string]$DetectorSamples)
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
Set-RunEnv "TRAIN_DATA_SOURCE_OVERRIDE" "random_ellipses"
Set-RunEnv "VAL_DATA_SOURCE_OVERRIDE" "random_ellipses"
Set-RunEnv "NOISE_MODE_OVERRIDE" "multiplicative"
Set-RunEnv "NOISE_LEVEL_OVERRIDE" $noiseText
Set-RunEnv "MODEL_ARCH_OVERRIDE" "tv_pc_cascade_unet"
Set-RunEnv "REFINER_INPUT_MODE_OVERRIDE" "u2_stacked"
Set-RunEnv "DATA_FIDELITY_CHANNEL_MODE_OVERRIDE" "stacked_selected"
Set-RunEnv "UNET_BACKBONE_OVERRIDE" "rad_unet"
Set-RunEnv "UNET_BASE_CHANNELS_OVERRIDE" "64"
Set-RunEnv "UNET_DEPTH_OVERRIDE" "4"
Set-RunEnv "UNET_RESIDUAL_MAX_OVERRIDE" "0.25"
Set-RunEnv "PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE" "1"
Set-RunEnv "PHYSICS_RESIDUAL_MODE_OVERRIDE" "stacked_selected_cg"
Set-RunEnv "PHYSICS_RESIDUAL_DAMPING_OVERRIDE" "1e-2"
Set-RunEnv "PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE" "8"
Set-RunEnv "PHYSICS_RESIDUAL_DETACH_OVERRIDE" "1"
Set-RunEnv "PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE" "1"
Set-RunEnv "PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE" "1"
Set-RunEnv "PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE" "0.13"
Set-RunEnv "PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE" "0.25"
Set-RunEnv "PHYSICS_GATE_MODE_OVERRIDE" "spatial"
Set-RunEnv "REFINER_STAGES_OVERRIDE" "2"
Set-RunEnv "REFINER_SHARE_WEIGHTS_OVERRIDE" "1"
Set-RunEnv "REFINER_STAGE_DC_ENABLED_OVERRIDE" "1"
Set-RunEnv "REFINER_STAGE_DC_CG_ITERS_OVERRIDE" "4"
Set-RunEnv "REFINER_STAGE_DC_DAMPING_OVERRIDE" "1e-2"
Set-RunEnv "REFINER_STAGE_DC_DETACH_OVERRIDE" "1"
Set-RunEnv "REFINER_STAGE_DC_NORMALIZE_OVERRIDE" "1"
Set-RunEnv "BASE_LR_OVERRIDE" "3e-4"
Set-RunEnv "LR_SCHEDULE_OVERRIDE" "constant_cosine"
Set-RunEnv "LR_CONSTANT_STEPS_OVERRIDE" "4500"
Set-RunEnv "LR_MIN_FACTOR_OVERRIDE" "0.3"
Set-RunEnv "LR_WARMUP_STEPS_OVERRIDE" "100"
Set-RunEnv "SCALAR_LR_RATIO_OVERRIDE" "1.0"
Set-RunEnv "INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE" "1"
Set-RunEnv "INTERMEDIATE_SUPERVISION_WEIGHT_START_OVERRIDE" "0.4"
Set-RunEnv "INTERMEDIATE_SUPERVISION_WEIGHT_END_OVERRIDE" "1.0"
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
Set-RunEnv "RESULTS_DIR_OVERRIDE" $resultDir
Set-RunEnv "MODEL_DIR_OVERRIDE" $checkpointDir
Set-RunEnv "CHECKPOINT_DIR_OVERRIDE" $checkpointDir
Set-RunEnv "MODEL_PATH_OVERRIDE" (Join-Path $checkpointDir ("theoretical_ct_{0}_model.pth" -f $outputTag))
Set-RunEnv "BEST_MODEL_PATH_OVERRIDE" (Join-Path $checkpointDir ("theoretical_ct_{0}_best_model.pth" -f $outputTag))

Write-Host "[run] project=$ProjectRoot"
Write-Host "[run] python=$PythonBin"
Write-Host "[run] K=$K detector_samples=$DetectorSamples measurements=$($K * $DetectorSamples)"
Write-Host "[run] alpha_json=$alphaJson"
Write-Host "[run] online_data=$onlineData offline_train='$offlineTrain' offline_val='$offlineVal'"
Write-Host "[run] init=l2_tv_admm morozov=constrained radius=rms admm=80 cg=30 tol=1e-4"
Write-Host "[run] model=tv_pc_cascade_unet refiner=u2_stacked backbone=rad_unet stages=2 gate=spatial stage_dc=1"
Write-Host "[run] lr=3e-4 schedule=constant_cosine constant_steps=4500 min_factor=0.3 warmup=100 scalar_lr_ratio=1.0"
Write-Host "[run] output_tag=$outputTag"
Write-Host "[run] log_dir=$logDir"
Write-Host "[run] result_dir=$resultDir"
Write-Host "[run] checkpoint_dir=$checkpointDir"

if ($DryRun) {
    Write-Host "[run] DryRun enabled; environment prepared but training was not started."
    exit 0
}

& $PythonBin (Join-Path $ProjectRoot "models\deep_learn\train.py")
exit $LASTEXITCODE
