param(
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

$common = Join-Path $PSScriptRoot "run_k24_improve_common.ps1"
$params = @{
    Experiment = "run1_1stage_nodc_rmax005"
    Mode = $Mode
    NoiseLevel = $NoiseLevel
    OutputTag = $OutputTag
    OfflineDataDir = $OfflineDataDir
    OfflineTrainDataset = $OfflineTrainDataset
    OfflineValDataset = $OfflineValDataset
    OfflineEvalDataset = $OfflineEvalDataset
    ModelChoice = $ModelChoice
    ModelPath = $ModelPath
    NumSamples = $NumSamples
    OfflineBatchSize = $OfflineBatchSize
    ResultPrefix = $ResultPrefix
    ResultDir = $ResultDir
    ProjectRoot = $ProjectRoot
    PythonBin = $PythonBin
}
if ($AllowOnlineFallback) { $params.AllowOnlineFallback = $true }
if ($UseSheppLoganTestAsVal) { $params.UseSheppLoganTestAsVal = $true }
if ($DryRun) { $params.DryRun = $true }

& $common @params
exit $LASTEXITCODE
