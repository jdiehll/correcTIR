import os
import sys

relative_path_to_correcTIR_code =os.path.join(os.path.dirname(__file__), '../..', 'correcTIR')
absolute_path = os.path.abspath(relative_path_to_correcTIR_code)
print(absolute_path)
# os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'correcTIR')