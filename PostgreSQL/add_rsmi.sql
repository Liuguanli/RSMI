/* Load the kd extension */
/*---------------------- */

load '$libdir/spgrsmitreeproc.o';


CREATE OR REPLACE FUNCTION spg_rsmi_config(internal) RETURNS INTERNAL AS '/home/research/postgresql-12.0/src/backend/access/spgist/spgrsmitreeproc.o' LANGUAGE 'C';





CREATE OR REPLACE FUNCTION kdpoint_consistent(internal,internal,internal,internal,internal,internal) RETURNS internal AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_penalty(internal,internal,internal,internal,internal,internal,internal,internal,internal,internal) RETURNS internal AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_CheckInternalSplit(internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal) RETURNS internal AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_pickSplit(internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal) RETURNS internal AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_check(internal,internal,internal,internal,internal) RETURNS INTERNAL AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_childbp(internal,internal,internal,internal) RETURNS INTERNAL AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_keylen(internal) RETURNS INTERNAL AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_getPred(internal,internal) RETURNS internal AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_equals_op(point,point) RETURNS bool AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_inside_op(point,box) RETURNS bool AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kd_getparam(int) RETURNS int AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kdpoint_NN_consistent(internal,internal,internal,internal,internal,internal, internal,internal) RETURNS internal AS '$libdir/spgist-kd.so' LANGUAGE 'C';

CREATE OR REPLACE FUNCTION kd_check_chaining(internal,internal) RETURNS internal AS '$libdir/spgist-kd.so' LANGUAGE 'C';


/* Load the operators */
/* Operator == is for exact match */
/* Operator @@ is for NN search*/
/* Operator ^^ is for range search */
/*--------------------------------*/

DROP OPERATOR == (point,point) CASCADE;

DROP OPERATOR @@ (point,point) CASCADE;

DROP OPERATOR ^^ (point,box) CASCADE;

CREATE OPERATOR == ( LEFTARG = point, RIGHTARG = point, PROCEDURE = kdpoint_equals_op, RESTRICT = eqsel, JOIN = eqjoinsel);

CREATE OPERATOR @@ ( LEFTARG = point, RIGHTARG = point, PROCEDURE = kdpoint_equals_op, RESTRICT = eqsel, JOIN = eqjoinsel);

CREATE OPERATOR ^^ ( LEFTARG = point, RIGHTARG = box, PROCEDURE = kdpoint_inside_op, RESTRICT = contsel, JOIN = contjoinsel);

CREATE OPERATOR CLASS spgist_kdpoint_ops FOR TYPE point USING spgist AS OPERATOR 1 ==, OPERATOR 2 ^^ (point,box), OPERATOR 100 @@, FUNCTION 1 kdpoint_consistent(internal,internal,internal,internal,internal,internal), FUNCTION 2 kdpoint_penalty(internal,internal,internal,internal,internal,internal,internal,internal,internal,internal), FUNCTION 3 kdpoint_checkinternalsplit(internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal), FUNCTION 4 kdpoint_pickSplit(internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal,internal), FUNCTION 5 kdpoint_check(internal,internal,internal,internal,internal), FUNCTION 6 kdpoint_getPred(internal,internal), FUNCTION 7 kdpoint_childbp(internal,internal,internal,internal), FUNCTION 8 kdpoint_keylen(internal), FUNCTION 9 kd_getparam(int), FUNCTION 10 kdpoint_NN_consistent(internal,internal,internal,internal,internal,internal, internal,internal), FUNCTION 11 kd_check_chaining(internal,internal) ;


/* Create an example table */
/*-------------------------*/
drop table  kd_points;
create table kd_points ( p point ,  num bigint);
create index SP_point_index on kd_points using spgist(p spgist_kdpoint_ops);
