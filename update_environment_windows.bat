

set ANACONDA_FOLDER="C:\Users\%USERNAME%\Anaconda3"

call %ANACONDA_FOLDER%\Scripts\activate.bat %root%

call conda env update -f environment_win.yml


