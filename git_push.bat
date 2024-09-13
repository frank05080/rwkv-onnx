@echo off
:: Navigate to the directory where the Git repository is located (optional)
cd C:\Users\guanzhong.chen\Documents\virtualbox_share\gitrepos\rwkv-onnx

git status

:: Stage all changes
git add .

:: Commit with the message "1"
git commit -m "1"

:: Push the changes to the remote repository
:loop
git push
if %errorlevel%==0 (
    echo Push succeeded!
) else (
    echo Push failed, retrying in 1 seconds...
    timeout /t 1 /nobreak
    goto loop
)

:: Pause to see the output before closing the terminal (optional)
pause