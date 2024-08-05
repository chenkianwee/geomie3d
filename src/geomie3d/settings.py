# ==================================================================================================
#
#    Copyright (c) 2024, Chen Kian Wee (chenkianwee@gmail.com)
#
#    This file is part of geomie3d
#
#    geomie3d is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geomie3d is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with py4design.  If not, see <http://www.gnu.org/licenses/>.
#
# ==================================================================================================
NDECIMALS = 6
ATOL = 1e-06
RTOL = 1e-06

def update_precision(new_ndecimals: int):
    """
    set the precision for the whole geomie3d environment. absolute tolerance(ATOL) & relative tolerance(RTOL) = 1/10**NDECIMALS

    Parameters
    ----------
    new_ndecimals: int
        the number of decimals to round off all geometry operations.

    """
    globals()['NDECIMALS'] = new_ndecimals
    globals()['ATOL'] = 1/10**new_ndecimals
    globals()['RTOL'] = 1/10**new_ndecimals

def show_all_settings():
    """
    show all the settings values.
    """
    print(f"NDECIMALS = {NDECIMALS}")
    print(f"ATOL = {ATOL}")
    print(f"RTOL = {RTOL}")
