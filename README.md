# orbit-determination
Asteroid orbit determination code using the Method of Gauss (```orbit_det.py```) and an experimental JAX AutoGrad (```autograd.py```) approach for comparison. Overall, the Method of Gauss approach is a lot more effective than AutoGrad since it fundamentally uses prior knowledge about the physical equations surrounding graviation while AutoGrad is simply brute forcing the optimization process.

```obs.txt``` contains observations of asteroid 2004 LJ1 made thorugh the Sierra Remote Observatory and the Central Washington University Observatory.
