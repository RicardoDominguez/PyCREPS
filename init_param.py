import highpol


# Indexes for the state variables being fed to GP, policy, cost function
dyni = [2, 4]      # GP inputs
dyno = [1, 2, 3]   # GP outputs
difi = [1, 2, 3]   # Variables trained by differences
scal = [1, 1]      # Input scaling (before being fed to GP)
ipol = [1]         # Policy inputs
icos = [3]         # Cost function inputs

# Algorithm parameters
eps = 3            # Relative entropy bound (lower -> more exploration)
K = 3              # Number of policy iterations
M = 10000          # Number of simulated rollouts
NinitRolls = 20    # Number of initial rollouts

# Policy parameters
hipol_mean = 2700;
hipol_sigma = # eye(pol.nX) (simroll.H) .* (deviation^2) (200)

# Parameters of the simulated rollout
# TODO: DELETE THE ONES NOT NEEDED OR RENAME THEM
simroll.max_sim_time = 20 # (in seconds)
simroll.dt = 0.5 # (in seconds)
simroll.initX = zeros(size(dyno))  # Initial system state
simroll.H = simroll.max_sim_time / simroll.dt # Horizon of sim rollout
simroll.target = 0 # Lowest amount of energy possible
simroll.timeInPol = 0 # 1 if the first input for the rollout policy is the
                       # simulation time
# Terminate simulation if iMaxVar variable is > than maxVar
simroll.useMaxVar = 1 # 1 to activate the option
simroll.iMaxVar = 1 # In this case x
simroll.maxVar = 5e4
simroll.iCost = icos

# Parameters for interacting with real system
#TODO: REMOVE UNNECESARY
load_data_fcn = @readDataSTMstudio
policy_folder = 'Controller/Src/'
log_folder = 'Controller/Log/'
UV_folder = 'Controller\MDK-ARM\MainPWM_CTS.uvprojx'
STMstudioLogExe = 'Controller\Rec\STMStudioRollout.exe'
archive_folder = 'archive/'
base_file_name = 'd1'
polSampleT = simroll.dt * 1000 # In ms
controllerMaxRollTime = simroll.max_sim_time * 1000 # Duration of rollout

# Previous dynamics knowledge
# TODO: REMOVE / ADAPT
prev_data = 'dyndata/'
use_prev_dyn_data = 0
dyndata_file = 'dynamics0.mat'
if use_prev_dyn_data
    load([prev_data, dyndata_file])
else
    X = [] Y = []
end

# Cost function
cost_fcn = @cost_max_var

# Parameters of the low level policy
pol.minU = 0           # Minimum control action
pol.maxU = 4200        # Maximum control action
pol.sample = @policy
pol.nX = simroll.H # Numer of data points in the X-axis lookup table
pol.lookupX = linspace(0, simroll.max_sim_time, pol.nX + 1)
pol.lookupX = pol.lookupX(2:end) # Look-up table X axis data points
pol.deltaX = pol.lookupX(2) - pol.lookupX(1) # Even spacing in look-up
                                              # table X axis
pol.controllerDeltaX = pol.deltaX * 1000 # Value of deltaX exported to controller
    #(different than pol.deltaX if diffent units are needed,
    # for instance s vs ms)

# Define high level policy
hipol = HighPol()
