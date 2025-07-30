import os
import shutil
import subprocess

import fire

"""
This module requires cnfgen and minisat installed and available in your command line path. 
For CNFgen, see https://massimolauria.net/cnfgen/ for installation instructions.
For minisat, a install with 'sudo apt-get install minisat' works (please adapt to your distro).
"""

def phase_transition_clauses(num_vars):
    """
    Returns the number of clauses at the phase transition (where
    the probability of the 3-CNF formula to be SAT is ~50%) for
    the given number of variables.
    Some values are hardcoded based on Selman et al 1996 and Crawford et al 1996
    and the remaining are calculated via Crawford et al's formula.

    References:
    SELMAN, Bart; MITCHELL, David G.; LEVESQUE, Hector J. Generating hard satisfiability problems. Artificial intelligence, v. 81, n. 1-2, p. 17-29, 1996.
    CRAWFORD, James M.; AUTON, Larry D. Experimental results on the crossover point in random 3-SAT. Artificial intelligence, v. 81, n. 1-2, p. 31-57, 1996.
    """
    phases = {  # vars->clauses -- hard-coded values based on previous research
        20: 91,
        50: 218,
        100: 431,
        150: 645,
        200: 854
    }

    if num_vars in phases:
        return phases[num_vars]

    # Crawford 1996's equation
    return int(4.258*num_vars + 58.26 * num_vars**(-2/3))


def kcnfgen(output, n, m, k=3):
    """
    Generates a random k-CNF with the specified parameters.
    This is just a wrapper on cnfgen randkcnf passing the proper parameters.
    :param output: dest_dir of where the files will be saved
    :param n: number of variables
    :param m: number of clauses
    :param k: number of literals per clause
    """
    #for i in range(num_formulas):
        # output = f'{dest_prefix}/{i}_{k}_{n}_{m}.cnf'
    subprocess.call(['cnfgen', '--output', output, 'randkcnf', str(k), str(n), str(m)])


def generate_instances(dest_dir, n, m, k=3, num_instances=100, initial_offset=0, accept_unsat=False, tmpfile='/tmp/candidate.cnf'):
    """
    Generates k-CNF instances according to the specified parameters.
    Calls minisat to check the satisfiability status, then names the file accordingly.
    :param dest_dir: where to save the instances
    :param n: number of variables
    :param m: number of clauses
    :param k: variables per clause
    :param num_instances: number of formulas to generate
    :param initial_offset: start saving instances at which number?
    :param accept_unsat: save unsat instances as well?
    :param tmpfile: temporary file name where the instance is tested for satisfiability
    """
    num_sat, num_unsat = 0, 0
    while num_sat + num_unsat < num_instances:
        print(f'calling cnfgen with n={n}, m={m}, k={k}')
        kcnfgen(tmpfile, n, m, k)

        # TODO collect statistics

        try:  # checks satisfiability with minisat
            print('calling minisat')
            subprocess.check_call(['minisat', tmpfile], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            if ex.returncode == 10:  # satisfiable
                num_sat += 1
                shutil.move(tmpfile, f'{dest_dir}/sat_{num_sat+initial_offset:05}_k{k}_v{n}_c{m}.cnf')

            elif ex.returncode == 20:  # unsatisfiable
                if accept_unsat:
                    num_unsat += 1
                    shutil.move(tmpfile, f'{dest_dir}/unsat_{num_unsat+initial_offset:05}_k{k}_v{n}_c{m}.cnf')

            else:
                raise RuntimeError(f"Unexpected minisat return code '{ex.returncode}' on {tmpfile}")


def phase_transition_neighborhood(where, n, num_instances, step=0.1, num_steps=10, k=3):
    """
    Generates kCNF instances in the neighborhood of the phase transition region.
    :param where: directory where the instances will be saved
    :param n: number of variables
    :param num_instances: number of instances
    :param step: increments on the phase transition ratio
    :param num_steps: how many values to test around the phase transition point (half to each direction)
    :param k: number of variables per clause
    """
    os.makedirs(where, exist_ok=True)

    dest_format = where + '/{}_v' + str(n) + '_c{}_r{:.3f}'  # onphase?, vars=n, clauses, ratio

    # on phase transition
    clauses_at_phase = phase_transition_clauses(n)
    ratio_at_phase = clauses_at_phase / n

    dest = dest_format.format('onphase', clauses_at_phase, ratio_at_phase)
    os.makedirs(dest, exist_ok=True)
    generate_instances(dest, n, clauses_at_phase, k, num_instances)

    for i in range(1, 1 + num_steps // 2):
        # overconstrained
        ratio_over = ratio_at_phase + i * step
        clauses_over = round(n * ratio_over)
        dest = dest_format.format('over', clauses_over, ratio_over)
        os.makedirs(dest, exist_ok=True)
        generate_instances(dest, n, clauses_over, k, num_instances)

        # underconstrained -- TODO avoid duplicated code from above
        ratio_under = ratio_at_phase - i * step
        clauses_under = round(n * ratio_under)
        dest = dest_format.format('under', clauses_under, ratio_under)
        os.makedirs(dest, exist_ok=True)
        generate_instances(dest, n, clauses_under, k, num_instances)


def paper_instances():
    """
    Generates the instances as used in the paper.
    """
    for v in range(10, 101, 10):    # 101 because i want 10, 20, ..., 100
        print(f"Generating instances with {v} variables...")
        phase_transition_neighborhood(f'instances/phase/v{v}', v, 1000)


if __name__ == '__main__':
    fire.Fire()
