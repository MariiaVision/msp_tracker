

set ANACONDA_FOLDER=C:\Users\%USERNAME%\Anaconda3

call %ANACONDA_FOLDER%\Scripts\activate.bat %root%

call conda activate gui_tracker

call python gui_tracker.py

