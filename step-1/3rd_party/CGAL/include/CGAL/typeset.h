// Copyright (c) 2014
// INRIA Saclay-Ile de France (France)
//
// This file is part of CGAL (www.cgal.org); you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3 of the License,
// or (at your option) any later version.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL: https://github.com/CGAL/cgal/blob/releases/CGAL-5.0-beta1/NewKernel_d/include/CGAL/typeset.h $
// $Id: typeset.h 530238d %aI Marc Glisse
// SPDX-License-Identifier: LGPL-3.0+
//
// Author(s)     : Marc Glisse

#ifndef CGAL_TYPESET_H
#define CGAL_TYPESET_H
#include <CGAL/config.h>
#include <type_traits>

// Sometimes using tuple just to list types is overkill (takes forever to
// instantiate).

namespace CGAL {
  template<class...> struct typeset;
  template<class H,class...U> struct typeset<H,U...> {
    typedef H head;
    typedef typeset<U...> tail;
    typedef typeset type;
    template<class X> using contains = typename
      std::conditional<
        std::is_same<H,X>::value,
        std::true_type,
        typename tail::template contains<X>
      >::type;
    template<class X> using add = typename
      std::conditional<
        contains<X>::value,
        typeset<H,U...>,
	typeset<H,U...,X>
      >::type;
  };
  template<> struct typeset<> {
    typedef typeset type;
    template<class X> using contains = std::false_type;
    template<class X> using add = typeset<X>;
  };

  template<class T1, class T2> struct typeset_union_ :
    typeset_union_<typename T1::template add<typename T2::head>::type, typename T2::tail>
  {};
  template<class T> struct typeset_union_ <T, typeset<> > : T {};

  template<class T1, class T2>
    struct typeset_intersection_ {
      typedef typename T1::head H;
      typedef typename typeset_intersection_<typename T1::tail,T2>::type U;
      typedef typename
	std::conditional<T2::template contains<H>::value,
	typename U::template add<H>::type, U>::type type;
    };
  template<class T>
    struct typeset_intersection_<typeset<>,T> : typeset<> {};

  template<class T1, class T2>
    using typeset_union = typename typeset_union_<T1,T2>::type;
  template<class T1, class T2>
    using typeset_intersection = typename typeset_intersection_<T1,T2>::type;
}
#endif
