$body = @{
    message = "test"
    session_id = $null
    module_context = "module-1-introduction"
} | ConvertTo-Json

$response = Invoke-WebRequest `
    -Uri "https://physical-ai-and-humanoid-robotics-b-iota-two.vercel.app/api/chat/query" `
    -Method POST `
    -Headers @{"Content-Type"="application/json"} `
    -Body $body `
    -TimeoutSec 30

Write-Host "Status Code: $($response.StatusCode)"
Write-Host "Response: $($response.Content)"
