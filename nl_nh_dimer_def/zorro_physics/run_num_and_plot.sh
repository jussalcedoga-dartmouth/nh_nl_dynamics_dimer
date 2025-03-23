# rm -rf data/ plots/
python3 get_zorro_theory.py
python3 plot_zorro_time_domain_and_spectrum.py ## optional. In case we want to see all Drive Freq traces individually
python3 get_zorro_plot.py 
python3 get_zorro_rotation.py
exec bash