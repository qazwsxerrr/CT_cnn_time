param(
    [ValidateSet("8", "16", "24")]
    [int]$K = 24,
    [int]$DetectorSamples = 0,
    [double]$NoiseLevel = 0.1,
    [string]$OutputTag = "",
    [ValidateSet("best", "final")]
    [string]$ModelChoice = "best",
    [string]$ModelPath = "",
    [int]$NumSamples = 500,
    [int]$OfflineBatchSize = 20,
    [string]$OfflineEvalDataset = "",
    [string]$ResultPrefix = "",
    [string]$ResultDir = "",
    [ValidateSet("random_ellipses", "shepp_logan")]
    [string]$TestDataSource = "shepp_logan",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$PythonBin = "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe",
    [switch]$ForceOnline,
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

function Resolve-KDDetectorSamples {
    param([Parameter(Mandatory = $true)] [int]$AngleCount, [int]$Requested)
    if ($Requested -gt 0) { return $Requested }
    if ($AngleCount -eq 8) { return 2048 }
    if ($AngleCount -eq 16) { return 1024 }
    if ($AngleCount -eq 24) { return 704 }
    throw "Unsupported K=$AngleCount."
}

function Resolve-DefaultOfflineDataset {
    param(
        [Parameter(Mandatory = $true)] [string]$Directory,
        [Parameter(Mandatory = $true)] [string]$ExactName,
        [Parameter(Mandatory = $true)] [string]$Pattern
    )
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

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
Set-Location -LiteralPath $ProjectRoot

$DetectorSamples = Resolve-KDDetectorSamples -AngleCount $K -Requested $DetectorSamples
if ($DetectorSamples -le 0) { throw "DetectorSamples must be positive." }
if ($NumSamples -le 0) { throw "NumSamples must be positive." }
if ($OfflineBatchSize -le 0) { throw "OfflineBatchSize must be positive." }
if ($PythonBin -match "[\\/]" -and -not (Test-Path -LiteralPath $PythonBin -PathType Leaf)) {
    throw "Python executable not found: $PythonBin"
}

$angleJsonByK = @{
    8 = "data\alpha8_tv\alpha_selected8_dopt_hard_gap_16_30.json"
    16 = "data\alpha16_tv\alpha_selected16_dopt_hard_gap_9_14.json"
    24 = "data\alpha24_tv\alpha_selected24_dopt_hard_gap_6_9.json"
}

$alphaJson = Join-Path $ProjectRoot $angleJsonByK[$K]
Require-File -Path $alphaJson -Message "Selected-angle JSON not found."

$cnnAngleIndices = (0..($K - 1)) -join ","
$noiseText = $NoiseLevel.ToString("G", [System.Globalization.CultureInfo]::InvariantCulture)
$offlineNoiseTag = $noiseText.Replace(".", "p").Replace("-", "m")
$noiseTag = "n" + $offlineNoiseTag
$outputTag = $OutputTag.Trim()
if ([string]::IsNullOrWhiteSpace($outputTag)) { $outputTag = "a${K}_d${DetectorSamples}_${noiseTag}_offline" }

$caseRoot = Join-Path $PSScriptRoot ("{0}_angle_det{1}" -f $K, $DetectorSamples)
$logDir = Join-Path $caseRoot "log"
$defaultResultDir = Join-Path $caseRoot "result"
$checkpointDir = Join-Path $caseRoot "checkpoints"
$offlineDataDir = Join-Path $caseRoot "offline_data"
$resolvedResultDir = $ResultDir.Trim()
if ([string]::IsNullOrWhiteSpace($resolvedResultDir)) { $resolvedResultDir = $defaultResultDir }
if (-not [System.IO.Path]::IsPathRooted($resolvedResultDir)) { $resolvedResultDir = Join-Path $ProjectRoot $resolvedResultDir }

$resolvedModelPath = $ModelPath.Trim()
if ([string]::IsNullOrWhiteSpace($resolvedModelPath)) {
    $modelFileKind = if ($ModelChoice -eq "best") { "best_model" } else { "model" }
    $defaultModelPath = Join-Path $checkpointDir ("theoretical_ct_{0}_{1}.pth" -f $outputTag, $modelFileKind)
    $compactModelPath = $defaultModelPath -replace "\.pth$", "_compact.pth"
    $resolvedModelPath = if (Test-Path -LiteralPath $compactModelPath -PathType Leaf) { $compactModelPath } else { $defaultModelPath }
} elseif (-not [System.IO.Path]::IsPathRooted($resolvedModelPath)) {
    $resolvedModelPath = Join-Path $ProjectRoot $resolvedModelPath
}

$resolvedResultPrefix = $ResultPrefix.Trim()
if ([string]::IsNullOrWhiteSpace($resolvedResultPrefix)) { $resolvedResultPrefix = $outputTag }

$fileSuffix = "alpha${K}_noise${offlineNoiseTag}_det${DetectorSamples}_edgewsubset"
$offlineEval = ""
if (-not $ForceOnline) {
    $offlineEval = $OfflineEvalDataset.Trim()
    if ([string]::IsNullOrWhiteSpace($offlineEval)) {
        if ($TestDataSource -eq "shepp_logan") {
            $offlineEval = Resolve-DefaultOfflineDataset -Directory $offlineDataDir -ExactName ("test500_shepp_logan_tvinit_{0}.pt" -f $fileSuffix) -Pattern ("test*_shepp_logan_tvinit_{0}.pt" -f $fileSuffix)
        } else {
            $offlineEval = Resolve-DefaultOfflineDataset -Directory $offlineDataDir -ExactName ("val500_random_ellipses_tvinit_{0}.pt" -f $fileSuffix) -Pattern ("val*_random_ellipses_tvinit_{0}.pt" -f $fileSuffix)
        }
    } elseif (-not [System.IO.Path]::IsPathRooted($offlineEval)) {
        $offlineEval = Join-Path $ProjectRoot $offlineEval
    }
}
$useOfflineEval = -not [string]::IsNullOrWhiteSpace($offlineEval)

New-Item -ItemType Directory -Force -Path $logDir, $resolvedResultDir, $checkpointDir | Out-Null

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
Set-RunEnv "DATA_SOURCE_OVERRIDE" $TestDataSource
Set-RunEnv "TEST_DATA_SOURCE_OVERRIDE" $TestDataSource
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
Set-RunEnv "OUTPUT_TAG_OVERRIDE" $outputTag
Set-RunEnv "LOG_DIR_OVERRIDE" $logDir
Set-RunEnv "RESULTS_DIR_OVERRIDE" $resolvedResultDir
Set-RunEnv "RESULT_DIR_OVERRIDE" $resolvedResultDir
Set-RunEnv "MODEL_LOAD_PATH_OVERRIDE" $resolvedModelPath
Set-RunEnv "MODEL_DIR_OVERRIDE" $checkpointDir
Set-RunEnv "CHECKPOINT_DIR_OVERRIDE" $checkpointDir

Write-Host "[test] project=$ProjectRoot"
Write-Host "[test] python=$PythonBin"
Write-Host "[test] K=$K detector_samples=$DetectorSamples measurements=$($K * $DetectorSamples)"
Write-Host "[test] alpha_json=$alphaJson"
Write-Host "[test] model_choice=$ModelChoice model=$resolvedModelPath"
Write-Host "[test] num_samples=$NumSamples test_data_source=$TestDataSource noise=multiplicative delta=$noiseText"
Write-Host "[test] offline_eval=$useOfflineEval offline_dataset='$offlineEval'"
Write-Host "[test] output_tag=$outputTag result_prefix=$resolvedResultPrefix"
Write-Host "[test] result_dir=$resolvedResultDir"

if ($DryRun) {
    $existsText = if (Test-Path -LiteralPath $resolvedModelPath -PathType Leaf) { "yes" } else { "no" }
    $offlineExistsText = if (-not [string]::IsNullOrWhiteSpace($offlineEval) -and (Test-Path -LiteralPath $offlineEval -PathType Leaf)) { "yes" } else { "no" }
    Write-Host "[test] DryRun enabled; model_exists=$existsText offline_dataset_exists=$offlineExistsText and evaluation was not started."
    exit 0
}

Require-File -Path $resolvedModelPath -Message "Model checkpoint not found. Use -ModelPath, -OutputTag, or -ModelChoice."

if ($useOfflineEval) {
    Require-File -Path $offlineEval -Message "Offline eval dataset not found. Use -OfflineEvalDataset, generate offline data, or pass -ForceOnline."
    & $PythonBin (Join-Path $ProjectRoot "models\deep_learn\evaluate_best_model_offline_val.py") `
        --model-path $resolvedModelPath `
        --offline-val $offlineEval `
        --num-samples $NumSamples `
        --batch-size $OfflineBatchSize `
        --result-dir $resolvedResultDir `
        --result-prefix $resolvedResultPrefix
    exit $LASTEXITCODE
}

& $PythonBin (Join-Path $ProjectRoot "models\deep_learn\test.py") `
    --model-path $resolvedModelPath `
    --num-samples $NumSamples `
    --result-dir $resolvedResultDir `
    --result-prefix $resolvedResultPrefix
exit $LASTEXITCODE
