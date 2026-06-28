param(
    [ValidateSet("8", "16", "24")]
    [int]$K = 24,
    [int]$DetectorSamples = 0,
    [double]$NoiseLevel = 0.1,
    [int]$TrainSamples = 8000,
    [int]$ValSamples = 500,
    [int]$TestSamples = 500,
    [int]$TrainBatchSize = 100,
    [int]$ValBatchSize = 100,
    [int]$TestBatchSize = 100,
    [int]$SeedOffset = 0,
    [string]$OutputDir = "",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$PythonBin = "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe",
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

function Assert-PositiveInt {
    param(
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [int]$Value
    )
    if ($Value -le 0) {
        throw "$Name must be positive."
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

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
Set-Location -LiteralPath $ProjectRoot

$DetectorSamples = Resolve-KDDetectorSamples -AngleCount $K -Requested $DetectorSamples
Assert-PositiveInt -Name "DetectorSamples" -Value $DetectorSamples
Assert-PositiveInt -Name "TrainSamples" -Value $TrainSamples
Assert-PositiveInt -Name "ValSamples" -Value $ValSamples
Assert-PositiveInt -Name "TestSamples" -Value $TestSamples
Assert-PositiveInt -Name "TrainBatchSize" -Value $TrainBatchSize
Assert-PositiveInt -Name "ValBatchSize" -Value $ValBatchSize
Assert-PositiveInt -Name "TestBatchSize" -Value $TestBatchSize
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

$noiseText = $NoiseLevel.ToString("G", [System.Globalization.CultureInfo]::InvariantCulture)
$noiseTag = $noiseText.Replace(".", "p").Replace("-", "m")
$caseDir = "{0}_angle_det{1}" -f $K, $DetectorSamples
$resolvedOutputDir = $OutputDir.Trim()
if ([string]::IsNullOrWhiteSpace($resolvedOutputDir)) {
    $resolvedOutputDir = Join-Path $PSScriptRoot (Join-Path $caseDir "offline_data")
} elseif (-not [System.IO.Path]::IsPathRooted($resolvedOutputDir)) {
    $resolvedOutputDir = Join-Path $ProjectRoot $resolvedOutputDir
}
New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null

$fileSuffix = "alpha${K}_noise${noiseTag}_det${DetectorSamples}_edgewsubset"
$trainPath = Join-Path $resolvedOutputDir ("train{0}_random_ellipses_tvinit_{1}.pt" -f $TrainSamples, $fileSuffix)
$valPath = Join-Path $resolvedOutputDir ("val{0}_random_ellipses_tvinit_{1}.pt" -f $ValSamples, $fileSuffix)
$testPath = Join-Path $resolvedOutputDir ("test{0}_shepp_logan_tvinit_{1}.pt" -f $TestSamples, $fileSuffix)
$measurementCount = $K * $DetectorSamples

Set-RunEnv "PROJECT_ROOT" $ProjectRoot
Set-RunEnv "PYTHONUTF8" "1"
Set-RunEnv "PYTHON_BIN" $PythonBin
Set-RunEnv "EXPERIMENT_PROFILE_OVERRIDE" "alpha_condition"
Set-RunEnv "ALPHA_CONDITION_TOP_K_OVERRIDE" ([string]$K)
Set-RunEnv "ALPHA_CONDITION_JSON_OVERRIDE" $alphaJson
Set-RunEnv "ALPHA_GRAM_CACHE_DIR_OVERRIDE" (Join-Path $ProjectRoot "data\alpha_gram_cache")
Set-RunEnv "NUM_ANGLES_TOTAL_OVERRIDE" ([string]$K)
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
Set-RunEnv "NOISE_MODE_OVERRIDE" "multiplicative"
Set-RunEnv "NOISE_LEVEL_OVERRIDE" $noiseText

Write-Host "[offline] project=$ProjectRoot"
Write-Host "[offline] python=$PythonBin"
Write-Host "[offline] K=$K detector_samples=$DetectorSamples measurements=$measurementCount sampling=shifted_lattice_edge_weighted_subset"
Write-Host "[offline] alpha_json=$alphaJson"
Write-Host "[offline] init=l2_tv_admm morozov=constrained radius=rms admm=80 cg=30 tol=1e-4"
Write-Host "[offline] noise=multiplicative delta=$noiseText"
Write-Host "[offline] train=$TrainSamples val=$ValSamples test=$TestSamples"
Write-Host "[offline] batch train=$TrainBatchSize val=$ValBatchSize test=$TestBatchSize"
Write-Host "[offline] output_dir=$resolvedOutputDir"
Write-Host "[offline] train_output=$trainPath"
Write-Host "[offline] val_output=$valPath"
Write-Host "[offline] test_output=$testPath"

if ($DryRun) {
    Write-Host "[offline] DryRun enabled; environment prepared but dataset generation was not started."
    exit 0
}

& $PythonBin (Join-Path $ProjectRoot "models\data_genoration\offline_tvinit_data.py") `
    --train-val-test-splits `
    --num-angles $K `
    --alpha-json $alphaJson `
    --seed-offset $SeedOffset `
    --train-output $trainPath `
    --val-output $valPath `
    --test-output $testPath `
    --train-samples $TrainSamples `
    --val-random-ellipses-samples $ValSamples `
    --test-shepp-logan-samples $TestSamples `
    --train-batch-size $TrainBatchSize `
    --val-batch-size $ValBatchSize `
    --test-batch-size $TestBatchSize
exit $LASTEXITCODE
