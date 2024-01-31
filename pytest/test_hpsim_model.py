import hpsim as hps
import yaml

with open("hpsim_model.yml","r") as file_object:
    config=yaml.load(file_object,Loader=yaml.SafeLoader)

model = hps.create_hpsim_model(config)
model.evaluate({})

# plot results
beam = model.beam
mask = gmask = beam.get_good_mask()

print("*** Input Beam ***")
print("w/user units")
beam.print_results()

#sys.exit()
print("*** Starting Simulation ***\n")

# determine mask of particles used in analysis and plotting
wmask = beam.get_mask_with_limits('w', lolim = 0)
gmask = beam.get_good_mask(wmask)
mask = gmask

print("*** Output Beam ***")
print("w/user units")
beam.print_results(mask)

# create output plot
plot = hps.BeamPlot(nrow=4, ncol=3, hsize=16, vsize=12)
plot.iso_phase_space('xxp', beam, mask, 1)
plot.iso_phase_space('yyp', beam, mask, 2)
plot.iso_phase_space('phiw', beam, mask, 3 )
plot.hist2d_phase_space('xxp', beam, mask, 4)
plot.hist2d_phase_space('yyp', beam, mask, 5)
plot.hist2d_phase_space('phiw', beam, mask, 6)
plot.profile('x', beam, mask, 7, 'g-')
plot.profile('y', beam, mask, 8, 'g-')
plot.profile('phi', beam, mask, 9, 'g-')
plot.profile('xp', beam, mask, 10, 'g-')
plot.profile('yp', beam, mask, 11, 'g-')
plot.profile('w', beam, mask, 12, 'g-')
plot.show()
plot.fig.savefig("benchmark.png")
#exit()

