# Records docs/demo/run.gif from a real run: `vhs docs/demo/run.tape` from the repo root.
Output docs/demo/run.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 520
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms

Hide
Type "export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear" Enter
Sleep 4s
Show
Type "python -m evaluation_harness list" Enter
Sleep 2s
Type "python -m evaluation_harness run --model keyword-sentiment --task sentiment" Enter
Sleep 3s
