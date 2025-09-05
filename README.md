# VQVC

## Abstract
Voice quality is an often overlooked aspect of speech with many
communicative functions. Voice quality conveys both paralinguistic 
and pragmatic information, such as signalling speaker
stance and aids in grounding. In this paper, we present VoiceQualityVC, 
a tool that can manipulate the voice quality of
both natural and synthesized speech using voice quality features 
including CPPS, H1–H2, and H1–A3. VoiceQualityVC
is a research tool for perceptual experiments into voice quality 
and UX experiments for voice design. We perform an objective evaluation 
demonstrating the control of these features
as well as subjective listening tests of the paralinguistic attributes 
of intimacy, valence, and investment. In these listening 
tests breathy voice was rated as more intimate and more
invested than modal voice and creaky voice was rated as less
intimate and less positive.

VQVC requires the following in order to work:
* Clone the VQVC repository ````git clone https://github.com/Hfkml/VQVC/````
* Navigate to the repository and run ````pip install -r requirements.txt````
* Download [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) and save it in the folder ````wavlm/````
* Download the pretrained G_VQVC.pth model from the Github release (on the right of your screen)
* Open VQVC.ipynb
* While there are differences between using CreakVC and VQVC, it can still be useful to watch [this video](https://play.kth.se/media/Show%20and%20Tell%20/0_hpyq9vy1) for instructions
