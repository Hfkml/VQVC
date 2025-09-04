# creapy &minus; a Python-based tool for the automatic detection of creak in conversational speech
<tt>creapy</tt> [ˈkɻiːpʰaɪ] is a tool to detect creak in speech signals. 

## Prerequisites
### Git
The most convenient way to work with <tt>creapy</tt> is to `clone` this repository using <tt>Git</tt>. Find a handy tutorial [here](https://rogerdudler.github.io/git-guide/).

### Python

<tt>creapy</tt> is written in the programming language <tt>Python 3</tt>. You can download the most recent version (as of 02/10/2023) for your operating system using the following links
- [Windows Installer (64-bit)](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe)
- [macOS 64-bit universal2 installer](https://www.python.org/ftp/python/3.10.10/python-3.10.10-macos11.pkg)

If you are a user of Windows make sure to check the mark **Add Python 3.10 to PATH** to add python to the PATH so that you can find the `python`-keyword that links to the `python`-executable globally in your terminal.
![](https://i.imgur.com/qfydlwD.png)

To check if you successfully installed Python open up a new Terminal. Press the Windows key, type `cmd` and hit enter. 

A new terminal window opens where you can type `python --version` which should give you the latest version of python 3.10 installed on your system.
![](https://i.imgur.com/A3fHfcS.png)

macOS-users can do the same thing in their terminal by typing `python3 --version` instead.

### JupyterLab
We recommend using <tt>JupyterLab</tt> which provides an interactive python environment and is also needed to open the demo-notebook at `examples/creapy_demo.ipynb`.
To install, type
```bash
pip install jupyterlab
```
in your terminal or
```bash
pip3 install jupyterlab
```
if you are using macOS. You can open jupyter lab in your terminal by typing
```bash
jupyter lab
```
## Installation

<tt>creapy</tt> will be available on PyPI once published.
<!-- 
```
pip install creapy
```
-->
Until then, you may clone the repository from git. 
<!-- For now as creapy is not yet published the git repository can be cloned from Git. -->

Before cloning you should navigate to your desired directory in the terminal like `C:/Users/myusername/<path_where_repo_will_be_cloned_into>`. With the following command you can clone the repository either with ssh
```bash
git clone git@gitlab.tugraz.at:speech/creapy.git
```
or html 
```bash
git clone https://gitlab.tugraz.at/speech/creapy.git
```

After cloning, a new folder should be present: `C:/Users/myusername/<path_to_creapy_repository>`.

To finally install <tt>creapy</tt> you need to navigate into the new folder using your terminal (The command `cd <folder>` **c**hanges the **d**irectory to a folder given with `<folder>`). Then, execute
```bash
pip install -e .
```

<!-- To check if the installation was succesfull you can try to run a python script or a jupyter notebook with:
```python
import creapy
``` -->
## Disclaimer

Please note that <tt>creapy</tt> will modify your hand-labelled `.TextGrid` files in an automated fashion. While the tool should generally only add additional tiers to your <tt>TextGrid</tt>, these new tiers might overpopulate quite soon, especially after messing with the tool a bit and processing whole folders. Make sure to **copy** your files beforhand to assure a backup and the originality of your precious files.

## Basic Usage
### Classifying an audio file

After you imported <tt>creapy</tt> with
```python
import creapy
```
you can classify creak on your own audio file calling the function `process_file` with the path to the audio- and respective TextGrid-file
```python!
X_test, y_pred, sr = creapy.process_file(
    audio_path='<path_to_your_audio_file>', 
    textgrid_path='<path_to_your_textgrid_file>')
```
<tt>creapy</tt> will add a new tier to the file at `textgrid_path` containing the detected creak intervals. **Note**: We recommend to work on a copy of your original TextGrid file.

### Choosing different models
Depending on the speaker you may choose another pre-trained model given by the paramter `gender_model`. Per default, <tt>creapy</tt> uses the model trained on the genders male and female (=`all`). This is an example of the classification using a model trained on female speakers only
```python!
X_test, y_pred, sr = creapy.process_file(
    audio_path='<path_to_your_audio_file>', 
    textgrid_path='<path_to_your_textgrid_file>',
    gender_model='female')
```
`process_file` returns a `tuple` where the first element (`X_test`) are the calculated feature values for each block, the second element is the calculated creak probability (`y_pred`) and the third element (`sr`) is the samplingrate in `Samples/s`. 

<!-- The calculated creak probability is shown in the following plot:

|![](examples/creapy_creak_probability_example.png 'creak probability')|
| - |
|*Creak probability over time (blue) and hand labelled intervals (red)*| -->

<!-- The function `get_time_vector` returns an array containing the timesteps for each block in seconds. -->

The <tt>TextGrid</tt> file that is saved to `textgrid_path` could look like this in Praat:

|![](examples/creapy_creak_example_praat.PNG 'creak probability')|
| :-: |
|*Modified TextGrid in Praat with new creak tier*|


### Classifying a folder containing multiple audio files
You can perform the classification on a whole folder using the function `process_folder`.
```python!
creapy.process_folder(
    audio_directory='<path_to_your_audio_directory>',
    textgrid_directory='<path_to_your_textgrid_directory>')
```

Here the folder structure is expected to be as follows:
```
audio_directory
├── speaker_1.wav
├── speaker_2.wav
├── ...
└── speaker_n.wav

textgrid_directory
├── speaker_1.TextGrid
├── speaker_2.TextGrid
├── ...
└── speaker_n.TextGrid
```

i.e. the maximum folder level is 1 and the filenames for the respective <tt>TextGrid</tt> files are expected to be the same as for the audio-files. **Note**: The audio directory and textgrid directory can be the same, but it is really important that the names and suffixes are correct.

## Set your configuration/Change parameters
All of the functions above use the default configuration. If you want to change various parameters you can do so in the configuration file. This file will be saved to `C:/Users/myusername/.creapy/config.yaml` after you **called the function** `set_config` **once**.

```python!
creapy.set_config()
```
A detailed description on the changeable parameters is given in the following table:

| Parameter | Explanation| default value |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- |
| `audio_directory`| Path to the directory containing audio files like `C:/Users/myusername/Documents/audio`.| `null`|
| `textgrid_directory` | Path to the directory containing the respective `TextGrid` files `C:/Users/myusername/Documents/textgrid`. This is optional. It will also work without texgrid files. |`null`|
| `csv_directory`| Path to a folder where the csv-files containing the classification results should be stored `C:/Users/myusername/Documents/results`.| `null`|
| `audio_start`| Starttime in seconds where the audiofile should be analysed. |$0$|
| `audio_end`| endtime in seconds until the audiofile should be analysed (if -1 the file gets processed until the end).|$-1$|
|`audio_suffix`| suffix of the audio-file(s) (compressed audio formats like `.mp3` are not supported). | `.wav`|
|`textgrid_suffix`|suffix of the textgrid-file(s).|`.TextGrid`|
|`gender_model`|The gender model chosen for creak-classification. Can be `all`, `male` or `female`. We recommend using the `all` model for male speakers and the `female` model for female speakers. |`all`|
|`tier_name`|The tiername of the new tier with <tt>creapy</tt>'s annotations in the textgrid file.| `creapy`|
|`block_size`| Classification blocksize in seconds. Smaller blocks are computationally more expensive but achieve a better time resolution. |$0.04$|
|`hop_size`|Classification hopsize in seconds. Should be in the range of $\frac{1}{4}\texttt{block\_size}\leq\texttt{hop\_size}\leq\frac{1}{2}\texttt{block\_size}$.|$0.01$|
|`creak_threshold`|Probability-threshold of the classifier where creak is classified. Can be a decimal value between 0 and 1. If you get too many false positives try increasing this value. If a lot of creaky voice gets missed try decreasing it.|$0.75$|
|`zcr_threshold`|Threshold for the zero-crossing-rate pre-elimination feature. Blocks with a $\text{zcr}\geq\text{zcr\_threshold}$ get eliminated. For female speakers we achieved better results with a higher value ranging around 0.10-0.18. For male speakers a value around 0.10-0.15 will yield good results. **Note:** This is highly speaker dependent. |$0.10$|
|`ste_threshold`|Threshold for the short-term-energy pre-elimination feature. This value does not normally need to be changed. It mostly eliminiates blocks of silence or noise. |$1\cdot10^{-5}$|

You can change these parameters in the `config.yaml` file itself or by using the function `set_config`, e.g.
```python!
creapy.set_config(block_size=0.05, creak_threshold=0.7)
```
If you want to reset the configuration file, you can do so, using the function `reset_config`, e.g.
```python!
creapy.reset_config()
```
This will set all parameters to their default values as seen in the table [above](#set-your-configurationchange-parameters).

Especially the `zcr_threshold`, `ste_threshold` and `creak_threshold` are parameters which can change the result the most. 

<!-- With the `plot` function those parameters get visualized. 
```python!
creapy.plot(X_test, y_pred, sr)
``` -->


### Plotting
While not a dependency for this tool, <tt>plotly</tt> is needed in order to evoke the plotting function provided by <tt>creapy</tt>:
```bash
pip install plotly
```
The function call
```python
creapy.plot(X_test, y_pred, sr)
```
on the example results in this plot
![](https://i.imgur.com/1nZUR8L.png)

The plot is interactive: one can check specific timeranges and select the features to display. In this way, a good estimate of the values for the parameters can be obtained.

<!-- ## Useful links (TEMP)

If you have a problem with setting up a SSH connection to GitHub the following tutorial should help: 

https://docs.github.com/en/authentication/connecting-to-github-with-ssh

We recommend working in a Jupyter enviroment. If you do not have one yet following link might help you:

https://jupyter.org/install

Python (brauch ma des?)

https://wiki.python.org/moin/BeginnersGuide/Download -->


<!-- On Windows, replace the Backslashes `\` or `\\` in your path with single slashes `/` so that you get something like `C:/Users/myusername/Documents/creak` -->

<!-- ### Classify audio file
You can use the pretrained models to detect creak in your speech signals. For more information on the training process, see [Pretrained Model](#pretrained-model).
1. change paths as described [above](#set-your-configuration).
2. run script `xyz.py` and wait; note: takes approx. 3 times as long as the audio file on a computer with processor xyz, ...; so for longer sound files, you need to be patient -->

# How to cite 
Please cite the following paper if you use <tt>creapy</tt>.
```
@inproceedings{creapy2023,
  title={Creapy: A Python-based tool for the detection of creak in conversational speech},
  author={Paierl, Michael and R{\"o}ck, Thomas and Wepner, Saskia and Kelterer, Anneliese and Schuppler, Barbara},
  booktitle={20th International Congress on Phonetic Sciences},
  year={2023}
} 
```
