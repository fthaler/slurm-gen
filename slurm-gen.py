#!/usr/bin/env python

# Copyright (c) 2017, Felix Thaler
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
import argparse
import sys
import re
import hashlib
import warnings


def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def parse_id_range(s):
    if s.lower() == 'id':
        return ['${SLURM_ARRAY_TASK_ID}']
    else:
        raise ValueError('can not parse id')


def parse_num_range(s):
    # range with optional step
    p = re.compile(r'(\d+\.?\d*)-(\d+\.?\d*)(:([\\+-\\*/]?)(\d+\.?\d*))?')
    m = p.match(s)
    if m:
        first = int_or_float(m.group(1))
        last = int_or_float(m.group(2))
        if m.group(3) is not None:
            op = '+' if m.group(4) is '' else m.group(4)
            step = int_or_float(m.group(5))
        else:
            op = '+' if first <= last else '-'
            step = 1
        if op not in {'+', '-', '*', '/'}:
            raise ValueError('invalid operation, must be one of +, -, *, /')

        def stepop(x):
            n = 0
            nmax = 10000
            while min(first, last) <= x <= max(first, last) and n < nmax:
                yield x
                if op == '+':
                    x += step
                if op == '-':
                    x -= step
                if op == '*':
                    x *= step
                if op == '/':
                    x /= step
                n += 1
            if n == nmax:
                warnings.warn('cut range at {} elements'.format(n))
        return list(stepop(first))
    else:
        raise ValueError('can not parse range')


def parse_list_range(s):
    # comma-separated list
    p = re.compile(r'[^,]+')
    l = p.findall(s)
    if not l:
        raise ValueError('can not parse list')
    return l


def parse_range(s):
    try:
        return parse_id_range(s)
    except ValueError:
        pass
    try:
        return parse_num_range(s)
    except ValueError:
        pass
    try:
        return parse_list_range(s)
    except ValueError:
        pass
    raise ValueError('could not parse range {}'.format(s))


def get_ranges(argstr):
    groups = dict()
    ranges = []

    def subf(match):
        range_id = match.group(2)
        range_str = match.group(3)
        if range_str:
            idx = len(ranges)
            if range_id:
                if range_id in groups:
                    raise ValueError('group with ID {} defined multiple times'
                                     .format(range_id))
                groups[range_id] = idx
            ranges.append(range_str)
        else:
            if range_id not in groups:
                raise ValueError('group with ID {} not defined'
                                 .format(range_id))
            idx = groups[range_id]
        return '${{v{}}}'.format(idx)

    p = re.compile(r'\[(([^=]+)=)?([^\]]*)\]')
    argstr = p.sub(subf, argstr)

    return argstr, ranges


def generate_sbatch(argstr, slurm_options, env_options,
                    slurm_output, outfile, parallel_limit,
                    verbose):
    inargstr = argstr
    argstr, ranges = get_ranges(argstr)

    range_values = [parse_range(r) for r in ranges]

    total_jobs = 1
    for r in range_values:
        total_jobs *= len(r)

    if verbose:
        print('total number of jobs to run:', file=sys.stderr)
        print('  {}'.format(total_jobs), file=sys.stderr)
        print('translated ranges:', file=sys.stderr)
        for r, vs in zip(ranges, range_values):
            vstrs = [str(v) for v in vs]
            if len(vs) > 10:
                vstr = '  {:20} -> {}, ..., {}'.format(r,
                                                       ', '.join(vstrs[:5]),
                                                       ', '.join(vstrs[-5:]))
            else:
                vstr = '  {:20} -> {}'.format(r, ', '.join(vstrs))
            print(vstr, file=sys.stderr)
        if slurm_options:
            print('additional options passed to slurm:', file=sys.stderr)
            print('  ' + ' '.join('--' + opt for opt in slurm_options),
                  file=sys.stderr)
        if env_options:
            print('additional environment variables set:', file=sys.stderr)
            print('  ' + ' '.join(env_options))

    if slurm_output is None:
        # generate hashed output name from app arguments
        h = hashlib.md5()
        h.update(argstr)
        slurm_output = h.hexdigest()

    # generate header code

    print('#!/bin/bash -l', file=outfile)
    for opt in slurm_options:
        print('#SBATCH --{}'.format(opt), file=outfile)

    if parallel_limit:
        if parallel_limit <= 0:
            raise ValueError('parallel job limit must be positive')
        print('#SBATCH --array=0-{}%{}'.format(total_jobs - 1, parallel_limit),
              file=outfile)
    else:
        print('#SBATCH --array=0-{}'.format(total_jobs - 1), file=outfile)
    print('#SBATCH --output={}_%a.out'.format(slurm_output), file=outfile)
    print(file=outfile)

    print('# sbatch script generated by slurm-gen using arguments:',
          file=outfile)
    print('# ' + inargstr, file=outfile)
    print(file=outfile)

    # generate array code

    for i, vs in enumerate(range_values):
        print('varray{}=({})'.format(i, ' '.join(str(v) for v in vs)),
              file=outfile)
    print(file=outfile)

    # generate index computation code

    print('r=${SLURM_ARRAY_TASK_ID}', file=outfile)
    for i, vs in enumerate(range_values):
        print('d=$(($r/{}))'.format(len(vs)), file=outfile)
        print('i{}=$(($r - $d*{}))'.format(i, len(vs)), file=outfile)
        print('r=$d', file=outfile)
        print('v{0}=${{varray{0}[${{i{0}}}]}}'.format(i), file=outfile)
        print(file=outfile)

    # generate application invocation code

    print(('if [ ! -s "{0}_${{SLURM_ARRAY_TASK_ID}}.out" ] || '
           '[ -n "$(grep -l \'srun: error\' '
           '"{0}_${{SLURM_ARRAY_TASK_ID}}.out")" ]').format(slurm_output),
          file=outfile)
    print('then', file=outfile)
    print('    {} srun '.format(' '.join(env_options)) + argstr, file=outfile)
    print('fi', file=outfile)


def main():
    parser = argparse.ArgumentParser(
        usage='%(prog)s [options] -- [application] [application arguments]',
        description=(
            'This script generates a slurm array job script '
            'for starting the given application with the given arguments.\n'
            'The resulting sbatch file is printed to stdout.\n'
            '\n'
            'Arguments given in brackets are interpreted as ranges of values '
            'and lead to multiple executions of the given application.\n'
        ),
        epilog=(
            'range examples:\n'
            '  [1-5]     -> 1, 2, 3, 4, 5\n'
            '  [1-5:1]   -> 1, 2, 3, 4, 5\n'
            '  [1-5:+1]  -> 1, 2, 3, 4, 5\n'
            '  [1-5:+2]  -> 1, 3, 5\n'
            '  [5-1]     -> 5, 4, 3, 2, 1\n'
            '  [5-1:-2]  -> 5, 3, 2\n'
            '  [2-8:*2]  -> 2, 4, 8\n'
            '  [8-2:/2]  -> 8, 4, 2\n'
            '  [0.1-0.3] -> 0.1, 0.2, 0.3\n'
            '  [foo,bar] -> foo, bar\n'
            '\n'
            'range groups:\n'
            '  Range groups allow to use the same range in several places:\n'
            '  [0=1-3] creates group 0 with values 1, 2, 3\n'
            '  [0=]    references group 0 with values 1, 2, 3\n'
            '\n'
            'range syntax:\n'
            "  RANGE := '[' [ GROUP ] ID | NUMRANGE | LIST ']'\n"
            "  GROUP := INT '='\n"
            "  ID := 'id'\n"
            "  NUMRANGE := NUM '-' NUM [ ':' STEP ]\n"
            "  STEP := [ ( '+' | '-' | '*' | '/' ) ] NUM\n"
            "  LIST := LISTITEM [ ',' LIST ]\n"
            '\n'
            'full examples:\n'
            '  create sbatch file for 30 runs of ./app, '
            'with argument x = 1, ..., 10, y = 3, 4, 5 and z = 1,\n'
            '  export OMP_NUM_THREADs=4 and run at max 4 jobs in parallel:\n'
            '    %(prog)s -e OMP_NUM_THREADs=4 -l 4 -- '
            './app --x [1-10] -y [3-5] -z 1\n'
            '\n'
            '  create sbatch file for 5 runs of ./app, '
            'with argument x = y = 1, 2, 4, 8, 16, z = linear index\n'
            '    %(prog)s -- ./app --x [0=1-16:*2] --y [0=] -z [id]\n'
            '\n'
            '  create sbatch file for 6 runs of ./app, '
            'with argument a = foo, bar, b = 2, 1, 0\n'
            '    %(prog)s -- ./app -a [foo,bar] -b [2-0]\n'
        ), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--slurm', '-s', action='append',
                        default=[], metavar='ARGUMENT=VALUE',
                        help=('slurm arguments that should be added\n'
                              'example: %(prog)s -s time=02:00:00 '
                              '-s constraint=GPU -- ./app'))
    parser.add_argument('--environment', '-e', action='append',
                        default=[], metavar='VARIABLE=VALUE',
                        help=('environment variables to define\n'
                              'example: %(prog)s -e OMP_NUM_THREADS=16 '
                              '-e OMP_PROC_BIND=spread -- ./app'))
    parser.add_argument('--limit', '-l', type=int, metavar='N',
                        default=None,
                        help=('maximum number of array jobs to '
                              'run in parallel.\nexample: %(prog)s -l '
                              '10 -- ./app -x [1-1000]'))
    parser.add_argument('--output', '-o', metavar='FILE',
                        help='output sbatch file')
    parser.add_argument('--slurm-output', '-u', metavar='BASENAME',
                        help=('basename for slurm output files, '
                              'full name will be BASENAME_ID.out'))
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='enable verbose output')

    # split script and application arguments
    try:
        argstart = sys.argv.index('--')
        if argstart >= len(sys.argv) - 1:
            raise ValueError()
    except ValueError:
        parser.print_help()
        sys.exit()

    # parse script arguments
    argsmap = parser.parse_args(sys.argv[1:argstart])

    # concat application arguments
    argstr = ' '.join(sys.argv[argstart + 1:])

    if argsmap.output:
        outfile = open(argsmap.output, 'w')
    else:
        outfile = sys.stdout

    generate_sbatch(argstr, argsmap.slurm, argsmap.environment,
                    argsmap.slurm_output, outfile, argsmap.limit,
                    argsmap.verbose)

    if argsmap.output:
        outfile.close()


if __name__ == '__main__':
    main()
