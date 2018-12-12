from agents import *


def SimpleReflexAgentProgram():
    """This agent takes action based solely on the percept"""
    def program(percept):
        loc, status = percept
        return ('Suck' if status == 'Dirty'
                else 'Right' if loc == loc_A else 'Left')
    return program

#Only two squares in the vaccum environment
loc_A = (0, 0)
loc_B = (1, 0)
loc_dict = loc_A, loc_B
# Program
program = SimpleReflexAgentProgram()
simple_agent = Agent(program)
# Initialize the two-state environment
trivial_vacuum_env = TrivialVacuumEnvironment()
print("State of the Environment: {}.".format(trivial_vacuum_env.status))
trivial_vacuum_env.add_thing(simple_agent)
print("SimpleReflexVacuumAgent is located at {}.".format(simple_agent.location))
awards = 0

for i in range(1000):
    # Run
    trivial_vacuum_env.step()

    for loc in loc_dict:
        if trivial_vacuum_env.status[loc] == 'Clean': awards += 1
        print("performance score is " + str(awards))

    # Check the current state of the environment
    print("State of the Environment: {}.".format(trivial_vacuum_env.status))
    print("SimpleReflexVacuumAgent is located at {}.".format(simple_agent.location))
