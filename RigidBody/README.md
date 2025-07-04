# There are the following scripts available:

## Typical workflow (samples initial conditions, simulates, learns, simulates learned, and compares):

    python3 comparison.py --generate --steps=100 --implicit --soft --without --model=RB
    python3 plot_compare.py --plot_L_errors --plot_m_errors --plot_msq_errors

## It is also possible to compare just training an validation losses 

    python3 compare_train_errors.py

## Alternatively, you can do that step by step
First generate dataset for training with:

    python3 simulate.py --generate --steps=50000 --model=RB

for the rigid body (or HT for heavy top, or P3D for the particle in three dimensions)

Then we train implicit and soft networks:

    python3 learn.py --method=without --model=RB

(or implicit or soft).

Then choose a different initial conditions and see how well our network fits the evolution. If the initial condition is too different we will not get a good fit. If it is the same we will fit perfectly. 

    python3 simulate.py --steps=500 --generate
    python3 simulate.py --steps=500 --implicit
    python3 simulate.py --steps=500 --soft
    python3 simulate.py --steps=500 --without

## And we can plot and see:

    python3 plot_compare.py --plot_m --plot_E --plot_L

Check training error and errors while learning. 
