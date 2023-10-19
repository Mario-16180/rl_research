import argparse

parser = argparse.ArgumentParser(description='Process some integers.') # Describes the program
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator') # Adds a positional argument
parser.add_argument('-single_word_message', '--swm', metavar='M', type=str, nargs=1,)
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)') # Explain each attribute: dest, action, const, default, help
# dest: the name of the attribute to be added to the object returned by parse_args()
# action: the basic type of action to be taken when this argument is encountered at the command line
# const: the value to be associated with the attribute named by dest
# default: the value to be used when the command-line argument is absent
# help: a brief description of what the argument does
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='verbose output') # Adds an optional argument. This argument is a flag, meaning that it doesn't need a value. 
                                            # If the flag is present, args.verbose is True, otherwise it is False.
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
args = parser.parse_args() # This line parses the arguments and stores them in args
print(args.swm[0])
print(args.integers)
print(args.integers[0], type(args.integers[0]))
print(args.accumulate(args.integers))
# print(args.verbose)
# print(args.integers)
# print(args.version)