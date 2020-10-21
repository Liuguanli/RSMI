// -*- C++ -*-
//
//  $RCSfile: hilbert.H,v $
//     $Date: 2001/01/29 14:25:56 $
// $Revision: 1.1 $
//    $State: Exp $
//   $Author: jan $
//
//   $Source: /u/jan/tpie_0.9.01b/work/RTreeTools/include/RCS/hilbert.H,v $
//   $Locker:  $
// 
//  Description:     declarations for Hilbert values
//  Created:         02.02.1998
//  Author:          Jan Vahrenhold
//  mail:            jan.vahrenhold@math.uni-muenster.de
//
//  Copyright (C) 1998 by  
// 
//  Jan Vahrenhold
//  Westfaelische Wilhelms-Universitaet Muenster
//  Institut fuer Informatik
//  Einsteinstr. 62
//  D-48149 Muenster
//  GERMANY
//

#ifndef HILBERT_H
#define HILBERT_H

#include <stdlib.h>
#include "hilbert4.H"

long long compute_Hilbert_value(long long x, long long y, long long side);

// different hilbert value sequence against compute_Hilbert_value(long long, long long, long long);
bitmask_t compute_Hilbert_value(bitmask_t x[], const size_t& x_len, const long long& bits);
long long compute_Hilbert_value(long long x[], const size_t& x_len, const long long& bits);

#endif
