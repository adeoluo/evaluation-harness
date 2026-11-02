# Records docs/demo/compare.gif from a real run: `vhs docs/demo/compare.tape` from the repo root.
Output docs/demo/compare.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m evaluation_harness run --model arithmetic-solver --task arithmetic` Enter
Sleep 2.5s
Type `python -m evaluation_harness run --model constant-zero --task arithmetic` Enter
Sleep 2.5s
