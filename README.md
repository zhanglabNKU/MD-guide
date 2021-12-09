A Two-level Dynamic Adaptive Network for Medical Image Fusion
====

Code of A Two-level Dynamic Adaptive Network for Medical Image Fusion for image fusion tasks, including medical image fusion (CT-MRI, PET-MRI), and multi-focus image fusion.


Requirements
-
pytorch ==1.3.1<br>
numpy = 1.18.5<br>
scipy == 1.2.1<br>

For testing
----
multi-focus fusion:<br>
`cd multi-focus` <br>
`python test_images.py`<br>

medical image fusion:<br>
* CT-MRI<br>
`cd medical/ct-mri`<br>
`python test_images.py`<br>

* PET-MRI<br>
`cd medical/mri-pet`<br>
`python test_images.py`<br>

outputs: the output of fusion results <br>

baseline: the code of state-of-the-art methods that used in this paper.






