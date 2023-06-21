# LocAndTrackHumans
Repozytorium zawierające pliki projektu inżynierskiego pod tytułem Lokalizacja i śledzenie sylwetki człowieka na sekwencjach wideo.
# Instrukcja
## Instalacja
Projekt był realizowany w środowisku Windows 10. Wszystkie instrukcje są napisane dla tego systemu operacyjnego. Wymagane jest zainstalowanie dowolnej wersji języka Python 3. W projekcie korzystano z wersji 3.9.7. Następnie należy zainstalować biblioteki numpy, pandas, opencv oraz matplotlib.
### Instalacja OpenPose według instrukcji ze [strony](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
1.	Zainstalować CMake GUI ze [strony](https://cmake.org/).
2.	Zainstalować Microsoft Visual Studio 2019 Community. Podczas instalacji należy wybrać opcję Desktop Development with C++.
3.	W celu wykorzystania GPU ze wsparciem CUDA zainstalować CUDA 11.1.1 oraz cuDNN 8.1.0
4.	Reszta wymagań powinna zostać pobrana przez CMake podczas instalacji.
5.	Pobrać lub sklonować repozytorium GitHub z [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
6.	Pobrać lub sklonować repozytorium z [pybind11](https://github.com/pybind/pybind11). W czasie instalacji napotkano błąd, przez który to repozytorium nie zostało pobrane automatycznie przez CMake. Pobrano je ręcznie i umieszczono w folderze openpose-master/3rdparty.
7.	W programie CMake jako folder z kodem źródłowym (source code) wybrać folder z OpenPose. Jako folder build wybrać dowolny inny folder lub stworzyć nowy folder build w openpose- master.
8.	Nacisnąć przycisk Configure i w polu Specify the generator for this project wybrać zainstalowaną wersję Visual Studio. W polu Optional platform for generator wybrać x64. Pole Optional toolset to use pozostawić puste. Nacisnąć Finish.
9.	Zaznaczyć flagę BUILD_PYTHON i nacisnąć Configure.
10.	W celu wykorzystania GPU ze wsparaciem CUDA, flagę GPU_MODE zmienić na CUDA.
11.	Jeśli na dole okna CMake widnieje napis Configuring done, oznacza to, że konfiguracja przebiegła poprawnie.
12.	Nacisnąć przycisk Generate.
13.	Nacisnąć przycisk Open Project. Spowoduje to otworzenie Visual Studio. W panelu Visual Studio zmienić konfigurację z Debug na Release.
14.	Z Build menu wybrać i nacisnąć Build solution.

Po wykonaniu tych kroków OpenPose został zainstalowany na komputerze. 
Wszystkie pliki w języku Python służące do lokalizacji i detekcji muszą zostać umieszczone w folderze /openpose-master/build/examples/tutorial_api_python 
lub w innym wybranym wcześniej folderze build.






### Instalacja SMPL-X
1.	W konsoli uruchomić komendę `pip install smplx[all]`
2.	Pobrać lub sklonować [repozytorium](https://github.com/vchoutas/smplx) z serwisu GitHub.
3.	W konsoli uruchomić komendę `python setup.py install`

### Konfiguracja rich-toolkit
1.	Pobrać lub sklonować [repozytorium](https://github.com/paulchhuang/rich_toolkit) z serwisu GitHub.
2.	Pobrać model SMPL-X ze strony i umieścić w folder ze rich-toolkit/body_models/smplx/
3.	Pobrać modele ciał SMPL-X (SMPL-X bodies) ze [strony](https://rich.is.tue.mpg.de). Wymagane modele należą do zbiorów train i test. Pobrane modele umieścić w folderze rich-toolkit/data/bodies/test lub train, w zależności od sekwencji.
4.	Pobrać skany i pliki kalibracyjne (Scans and calibration files) ze [strony](https://rich.is.tue.mpg.de). Następnie umieścić je w folderze rich-toolkit/data/scan_calibration

Sekwencje z bazy RICH, z których korzystano w projekcie, 
można pobrać instalując program Git Bash dla systemu Windows i uruchamiając udostępniony plik sekwencje.sh za pomocą komendy: `sh sekwencje.sh`


## Obsługa
W celu odtworzenia rezultatów projektu należy pobrać repozytorium lub sklonować repozytorium. 
Następnie pliki z folderu eval oraz plik mapping.json należy umieścić w folderze zawierającym pobrane wcześniej repozytorium rich-toolkit. 
Natomiast pliki z folderu openpose należy umieścić w folderze /openpose-master/build/examples/tutorial_api_python, 
z powodów opisanych w części instalacja. 
W celu wykonania lokalizacji oraz śledzenia sylwetki człowieka na sekwencjach wideo należy użyć następujących plików:
1.	DroneEstimatePoseVideo.py – pozwala na wykonanie lokalizacji i śledzenia dla sekwencji wideo z drona.
2.	RichEstimatePose.py – pozwala na wykonanie lokalizacji i śledzenia dla sekwencji wideo z bazy RICH. 
3.	RichEstimatePoseCameras.py – posiada tą samą funkcję, co RichEstimatePose.py, ale pozwala na iterację po ujęć z kamer dla określonej sytuacji. Dzięki temu nie ma potrzeby uruchamiania pliku RichEstimatePose.py dla każdej sekwencji po kolei.

Następnie nazwy plików z adnotacjami dla sekwencji z bazy RICH należy zmienić na nazwy odpowiadających im klatek za pomocą pliku renamePred.py.
W celu ewaluacji wyników dla sekwencji z drona należy użyć następujących plików:
1.	DronEval.py – pozwala na ewaluację wyników w metrykach AP i PCKh,
2.	DronEvalThresh.py – pozwala na wykreślenie wykresu zależności wyników od progu.

W celu ewaluacji wyników dla sekwencji z bazy RICH należy użyć następujących plików:
1.	RICHEvaluationRaw.py lub RICHEvaluationRawCameras.py – pozwalają na zapis do plików pkl danych ground truth oraz ewaluację. Przed zapisem do pliku, w folderze zawierającym rich-toolkit należy utworzyć następującą ścieżkę: …/results/{train lub test}/{nazwa sekwencji}/{numer_kamery}.
2.	RICHEvalFromFile.py – plik służy do ewaluacji wyników poprzez odczytanie danych z plików pkl utworzonych w poprzednim punkcie. Zaleca się utworzenie plików pkl, a następnie korzystanie RICHEvalFromFile.py, ponieważ znacznie przyspiesza to wczytywanie danych ground truth. Wyniki ewaluacji są zapisywane do pliku csv.
3.	RICHEvalThreshCams.py – pozwala na zapis uśrednionych wyników ewaluacji dla różnych progów do plików csv. Działa w analogiczny sposób jak plik RichEstimatePoseCameras.py, iterując po ujęciach z kamer.
4.	RICHEvalFromCSV.py – do jego działania wymagane są pliki csv utworzone przez RICHEvalFromFile.py. Oblicza on wyniki zawarte w tabelach 3,7 i 10. Następnie zapisuje je do plików csv.
5.	RICHthreshEvalFromCSV.py – pozwala na wykreślenie wykresu zależności wyników od progu.

Do wizualizacji wyników służą następujące pliki:
1.	visualizeVideoDrone.py – odtwarza sekwencję wideo z drona z naniesionymi punktami zlokalizowanymi przez OpenPose oraz punktami prawdziwymi,
2.	visualizeResultsRICH.py – wizualizuje wybraną klatkę z sekwencji z bazy RICH z naniesionymi punktami zlokalizowanymi przez OpenPose oraz punktami prawdziwymi.


