
BFIELD     = 1 # Tesla
RESOLUTION = 0.1 # cm
ECHARGE    = 3e-3 # Gev/(c*Tesla)
HITEFFICIENCY = 0.95
PLANEXMAX  = 50
PLANEZTHICKNESS = 1
PARTICLEMULTIPLICITY = 10

def momentum( omega ) : return ECHARGE*BFIELD/omega
def omega( momentum ) : return ECHARGE*BFIELD/momentum

class DetectorPlane:
    __slots__ = [ "_z" ]
    def __init__(self, z):
        self._z = z
    def z(self) : return self._z
    def x1(self): return -PLANEXMAX
    def x2(self): return +PLANEXMAX

class SensorPlane(DetectorPlane):
    __slots__ = [ "_resolution", "_alignmentbias" ]
    def __init__(self, z, misalignment=0.0):
        DetectorPlane.__init__(self,z)
        self._resolution   = RESOLUTION
        self._alignmentbias = misalignment
    def isSensor(self) : return True
    def resolution(self) : return self._resolution
    def dz(self): return 2.0
    def alignmentBias(self): return self._alignmentbias
    def setAlignmentBias(self,b): self._alignmentbias = b

class Detector:
    __slots__ = ["_planes"]
    def __init__( self, planes ):
        self._planes = planes
    def planes(self): return self._planes
    
def configureDetector( mode=1 ):
    numplanes = 6
    z1 = 40
    z2 = 190
    dz = (z2 - z1)/(numplanes-1)
    det = Detector(planes = [ SensorPlane(z1 + i*dz) for i in range(numplanes) ])

    # misalign one plane
    if mode==2: det.planes()[3].setAlignmentBias(-1.0)
    
    return det

def drawDetector( axis, detector ):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # Create a Rectangle patch
    for p in detector.planes():
        dz = p.dz()
        dx = p.x2()-p.x1()
        z1 = p.z()-0.5*dz
        rect = patches.Rectangle((z1,p.x1()), dz, dx, linewidth=1, edgecolor='r', facecolor='none')
        axis.add_patch(rect)
        axis.update_datalim( [(z1,p.x1()),(z1+dz,p.x2())] )
    axis.update_datalim( [(-10,0)] )

class State:
    __slots__ = ["_z","_parameters" ]
    def __init__(self, z, parameters):
        self._z = z
        self._parameters = parameters
    def propagate(self, newz ):
        self._parameters[0] += self._parameters[1]*(newz-self._z)
        self._z = newz
    def z(self) : return self._z
    def x(self) : return self._parameters[0]
    def tx(self) : return self._parameters[1]

class MeasuredState(State):
    __slots__ = ["_covmatrix"]

class Hit:
    __slots__ = ["_plane","_truestate","_x"]
    def __init__(self, plane, state, x):
        self._plane = plane
        self._truestate = state
        self._x = x
    def x(self) : return self._x
    def z(self) : return self.plane().z()
    def plane(self) : return self._plane
        
class Particle:
    __slots__ = ["_stateAtOrigin","_hits"]
    def __init__(self, stateAtOrigin, hits):
        self._stateAtOrigin = stateAtOrigin
        self._hits = hits
    def hits(self): return self._hits
    def stateAtOrigin(self): return self._stateAtOrigin

class Event:
    __slots__ = [ "_particles" ]
    def __init__( self, particles ) :
        self._particles = particles
    def hits(self) :
        return sorted([ hit for p in self._particles for hit in p.hits()  ], key = lambda h: h.z() )
    def particles(self):
        return self._particles
    
def generateParticle( detector, minnumhits ):
    import numpy as np
    import copy
    from numpy import random
    xmax  = 5    # mm
    txmax = 0.3  # rad
    x0  = xmax * (random.rand()-0.5)
    tx0 = txmax * (random.rand()-0.5)
    # for now, no scattering
    state = State( 0, np.array( [x0,tx0] ) )
    stateAtOrigin = copy.deepcopy(state)
    hits = []
    for p in detector.planes():
        if p.isSensor():
            state.propagate( p.z() )
            # generate the effect on inefficiency
            if random.rand()<HITEFFICIENCY:
                # generate effects of hit resolution and add the alignment bias
                dx = random.normal()*p.resolution() + p.alignmentBias()
                hits.append( Hit( p, state, state.x() + dx ) )
    if len(hits)>=minnumhits:
        return Particle(stateAtOrigin=stateAtOrigin,hits=hits)
    return None
    
def generateEvent( detector, meanmultiplicity=PARTICLEMULTIPLICITY):
    from numpy import random
    particles = []
    for i in range( random.poisson( meanmultiplicity ) ):
        p = generateParticle(detector, 5)
        if p: particles.append(p)
    return Event( particles = particles )
    
def drawEvent( event, detector ):
    x = [ hit.z() for hit in event.hits() ]
    y = [ hit.x() for hit in event.hits() ]
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(1,1)
    axis.scatter( x, y )
    drawDetector( axis, detector)
    return (fig, axis)

if __name__ == "__main__":
    det   = configureDetector()
    event = generateEvent(det,6)
    drawEvent(event, det)
    import matplotlib.pyplot as plt
    plt.show()
