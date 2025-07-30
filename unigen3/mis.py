#!/usr/bin/env python3

# Copyright (C) 2016-2018   Kuldeep S. Meel
# Copyright (C) 2018        Mate Soos
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; version 2
# of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.


from __future__ import with_statement  # Required in 2.5
from __future__ import print_function

import os
import sys
import time
import optparse

try:
    # if we're imported elsewhere, make local binaries find-able
    PATH = sys.modules[__name__].__file__
    PATH = PATH[:PATH.index("/mis.py")]
except:
    # otherwise, we're being run directly. Assume togmus & muser2 are available
    PATH = "."

usage = "usage: %prog [options] <input.cnf>"
desc = """Gives the minimal independent set of the cnf"""

def set_up_parser():
    parser = optparse.OptionParser(usage=usage, description=desc)
    parser.add_option("--timeout", metavar="TOUT", dest="timeout", type=int,
                      default=0, help="timeout for each iteration in seconds. Default: %default. 0 = no timeout")
    parser.add_option("-v", default=0, type=int,
            dest="verbosity", help="higher numbers for more verbosity. Default: %default")
    parser.add_option("--noclean", action="store_true", default=False,
            dest="noclean", help="don't clean up temporary files. Default: %default")
    parser.add_option("--out", dest="outputfile", type=str, default=None,
            help="Output file destination. Default: 'inputfile.ind'")
    parser.add_option("--maxiter", dest="maxiter", type=int, default=1,
                      help="up to 'max' number of minimal independent supports will be generated. Default: %default")
    parser.add_option("--useind", dest="useind", action="store_true",
                      default=False,
                      help="use independent support provided in input file")
    parser.add_option("--glucose", action="store_true", default=False,
            dest="glucose", help="Use glucose in muser2. Default: %default")
    parser.add_option("--muser2bin", type=str, default=PATH+"/muser2-dir/src/tools/muser2/muser2",
                      dest="bin", help="muser2 binary to use")

    return parser

def mis(inputfile=None, outputfile=None, useind=False, maxiter=1, use_glucose=False,
        muser2_bin=PATH+"/muser2-dir/src/tools/muser2/muser2", timeout=0, verbosity=0,
        noclean=False):
    if outputfile is None:
        outputfile = inputfile + ".ind"
    
    mytime = time.time()
    gmusFile = outputfile + '.gcnf'
    tempOutFile = outputfile + '.tcnf'

    togmus_cmd = "%s/togmus %s %s %s" % (PATH, inputfile, gmusFile, useind)
    if verbosity > 0:
        print("Running togmus: '%s'" % togmus_cmd)
    os.system(togmus_cmd)
    if verbosity > 0:
        print("togmus executed in %-3.2f s" % (time.time()-mytime))

    muser2_cmd = "%s -v 0 -grp -comp -%s -order 4 -T %s %s > %s" % (
            muser2_bin, "glucose" if use_glucose else "minisats",
            timeout, gmusFile, tempOutFile)
    if verbosity > 1:
        print("Using %s for muser2" % ("glucose" if use_glucose else "minisats"))
    
    indMap = set()
    min_indvars_len = None
    min_indvars = None
    # run maxiters iterations
    for i in range(maxiter):
        if verbosity > 1:
            print("Iteration %u" % (i+1))
        mutime = time.time()
        if verbosity > 0:
            print("Running muser2: '%s'" % muser2_cmd)
        os.system(muser2_cmd)
        if verbosity > 0:
            print("muser2 finished in %-3.2f s" % (time.time()-mutime))

        indvars = parseOutput(tempOutFile).strip().lstrip(" v ")
        indvars_len = len(indvars.split())-1
        if min_indvars_len is None:
            min_indvars_len = indvars_len
        if indvars not in indMap:
            if verbosity > 1:
                print("New indvars: %s" % indvars)
            indMap.add(indvars)
            if indvars_len <= min_indvars_len:
                if verbosity > 1:
                    print("Minimal")
                min_indvars = indvars
                min_indvars_len = indvars_len
            else:
                if verbosity > 1:
                    print("Not minimal")
        else:
            if verbosity > 1:
                print("Rediscovered indvars (consider reducing maxiter)")

    if verbosity > 1:
        print("finished %u iterations in %-3.2f" % (maxiter, time.time()-mytime))

    with open(outputfile, 'a') as f:
        f.write(min_indvars + "\n")
        if verbosity > 0:
            print("Wrote output to %s" % outputfile)

    if not noclean:
        if verbosity > 1:
            print("Deleting temporary files")
        os.unlink(tempOutFile)
        os.unlink(gmusFile)
    else:
        if verbosity > 1:
            print("noclean selected")
            print("Leaving temporary files:\n%s\n%s" % (tempOutFile, gmusFile))

    return min_indvars

def parseOutput(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        f.close()
        for line in lines:
            if (line.strip().startswith('v')):
                return line
        return ''

def main():
    starttime = time.time()
    parser = set_up_parser()
    (options, args) = parser.parse_args()

    if len(args) < 1:
        print("ERROR: you must give a CNF input file as a parameter")
        exit(-1)

    inputfile = args[0]

    indvars = mis(inputfile, options.outputfile, options.useind, options.maxiter,
        options.glucose, options.bin, options.timeout, options.verbosity, options.noclean)
    
    print("Finished in %-3.2f seconds" % (time.time()-starttime))
    print("num independent vars:", len(indvars.split())-1)
    print("** Copy-paste the following line in the top of your CNF for ApproxMC **")
    print("c ind %s" % indvars)

if __name__ == "__main__":
    main()
