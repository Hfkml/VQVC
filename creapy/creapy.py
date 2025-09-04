import creapy

X_test, y_pred, sr = creapy.process_file(textgrid_path='../data/out/creaky_vctk_whisper/p232_180.TextGrid', audio_path='../data/out/creaky_vctk.wav')
creapy.plot(X_test, y_pred, sr, words=[{'start': 0.5, 'end': 1.5, 'word': 'hello'}])