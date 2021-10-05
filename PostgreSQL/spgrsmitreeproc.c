/*-------------------------------------------------------------------------
 *
 * spgrsmitreeproc.c
 *	  implementation of rsmi tree over points for SP-GiST
 *
 *
 * Portions Copyright (c) 1996-2020, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * IDENTIFICATION
 *			src/backend/access/spgist/spgrsmitreeproc.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "access/spgist.h"
#include "access/spgist_private.h"
#include "access/stratnum.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/float.h"
#include "utils/array.h"
#include "utils/geo_decls.h"

Datum
	spg_rsmi_config(PG_FUNCTION_ARGS)
{
	/* spgConfigIn *cfgin = (spgConfigIn *) PG_GETARG_POINTER(0); */
	spgConfigOut *cfg = (spgConfigOut *)PG_GETARG_POINTER(1);

	cfg->prefixType = ANYOID; /* to contain the model*/
	// cfg->prefixType = FLOAT8ARRAYOID; /* to contain the model*/
	cfg->labelType = VOIDOID; /* we don't need node labels */
	cfg->canReturnData = true;
	cfg->longValuesOK = false;
	PG_RETURN_VOID();
}

static __uint128_t
compute_Z_value(Point *p, int bit_num)
{
	__uint128_t x = p->x * ((__uint128_t)1 << bit_num);
	__uint128_t y = p->y * ((__uint128_t)1 << bit_num);

	__uint128_t result = 0;
	__uint128_t mask = 1;
	for (int i = 0; i < bit_num; i++)
	{
		__uint128_t seed = mask;

		__uint128_t temp = seed & x;
		temp = temp << i;
		result += temp;

		temp = seed & y;
		temp = temp << (i + 1);
		result += temp;

		mask = mask << 1;
	}
	// elog(WARNING, "compute_Z_value z: %d", result);

	return result;
}

typedef struct SortedPoint
{
	Point *p;
	__uint128_t z_value;
	int index;
} SortedPoint;

typedef struct Model
{
	float8 a;
	float8 b;
	float8 sub_model_num;
	float8 cardinality;
	float8 bit_num;
	// float8 x1;
	// float8 x2;
	// float8 y1;
	// float8 y2;
	// Point high;
	// Point low;
	BOX *mbr;
} Model;

//  TODO  train model
static Model *
train_model(SortedPoint *sorted, int n)
{
	// elog(WARNING, "train_model n: %d ", n);
	Model *model = (Model *)palloc(sizeof(Model));
	model->a = (float8)(n - 1) / (sorted[n - 1].z_value - sorted[0].z_value);
	// elog(WARNING, "train_model a: %f ", model->a);
	// model->b = ((float8)(n - 1) - model->a * (sorted[n - 1].z_value + sorted[0].z_value)) / 2;
	model->b = -model->a * sorted[0].z_value;
	// elog(WARNING, "train_model b: %f ", model->b);
	return model;
}

//  predict  via model when building structure
static int
predict(Model *model, SortedPoint sorted)
{
	__uint128_t z_value = sorted.z_value;
	int predict_res = (model->a * z_value + model->b) * model->sub_model_num / model->cardinality;
	predict_res = predict_res < 0 ? 0 : predict_res;
	predict_res = predict_res >= model->sub_model_num ? model->sub_model_num - 1 : predict_res;
	elog(WARNING, "predict model_num: %f cardinality: %f predict_res: %d", model->sub_model_num, model->cardinality, predict_res);
	return predict_res;
}

//  predict via array
static int
predict_via_array(Model *model, Point *point)
{
	// elog(WARNING, "predict_via_array");

	// elog(WARNING, "predict_via_array a: %f b: %f model_num: %d cardinality:%d bit_num:%d", model->a, model->b, model->sub_model_num, model->cardinality, model->bit_num);

	__uint128_t z_value = compute_Z_value(point, model->bit_num);

	// elog(WARNING, "predict_via_array z_value: %d", z_value);

	int predict_res = (model->a * z_value + model->b) * model->sub_model_num / model->cardinality;
	predict_res = predict_res >= model->sub_model_num ? model->sub_model_num - 1 : predict_res;
	predict_res = predict_res < 0 ? 0 : predict_res;
	// elog(WARNING, "predict_via_array predict_res: %d", predict_res);

	return predict_res;
}

static int
z_value_cmp(const void *a, const void *b)
{
	SortedPoint *pa = (SortedPoint *)a;
	SortedPoint *pb = (SortedPoint *)b;

	if (pa->z_value == pb->z_value)
		return 0;
	return (pa->z_value > pb->z_value) ? 1 : -1;
}

// Second invoked
Datum
	spg_rsmi_choose(PG_FUNCTION_ARGS)
{
	spgChooseIn *in = (spgChooseIn *)PG_GETARG_POINTER(0);
	spgChooseOut *out = (spgChooseOut *)PG_GETARG_POINTER(1);
	Point *inPoint = DatumGetPointP(in->datum);
	// elog(WARNING, "spg_rsmi_choose in->allTheSame: %d", in->allTheSame);
	// elog(WARNING, "spg_rsmi_choose nNodes: %d", in->nNodes);

	if (in->allTheSame)
	{
		out->resultType = spgMatchNode;
		/* nodeN will be set by core */
		out->result.matchNode.levelAdd = 0;
		out->result.matchNode.restDatum = PointPGetDatum(inPoint);
		PG_RETURN_VOID();
	}

	Assert(in->hasPrefix);

	Model *model_res = (Model *)DatumGetPointer(in->prefixDatum);
	elog(WARNING, "model_res: %f, %f, %f, %f, %f", model_res->a, model_res->b, model_res->sub_model_num, model_res->cardinality, model_res->bit_num);

	// ArrayType *model_array = DatumGetArrayTypeP(in->prefixDatum);
	// elog(WARNING, "spg_rsmi_choose model_array->ndim: %d", model_array->ndim);
	// float8 *model_values = (float8 *)ARR_DATA_PTR(model_array);

	out->resultType = spgMatchNode;
	int nodeN = predict_via_array(model_res, inPoint);
	elog(WARNING, "spg_rsmi_choose nodeN: %d", nodeN);

	out->result.matchNode.nodeN = nodeN;
	out->result.matchNode.levelAdd = 1;
	out->result.matchNode.restDatum = PointPGetDatum(inPoint);

	PG_RETURN_VOID();
}

// First invoked
Datum
	spg_rsmi_picksplit(PG_FUNCTION_ARGS)
{
	elog(WARNING, "spg_rsmi_picksplit");
	spgPickSplitIn *in = (spgPickSplitIn *)PG_GETARG_POINTER(0);
	spgPickSplitOut *out = (spgPickSplitOut *)PG_GETARG_POINTER(1);
	int i;
	SortedPoint *sorted;

	sorted = palloc(sizeof(*sorted) * in->nTuples);

	int bit_num = ceil((log(in->nTuples)) / log(2));

	// elog(WARNING, "spg_rsmi_picksplit bit_num: %d", bit_num);
	// elog(WARNING, "spg_rsmi_picksplit in->nTuples: %d", in->nTuples);
	double inf = get_float8_infinity();

	float8 min_x = inf;
	float8 max_x = -inf;
	float8 min_y = inf;
	float8 max_y = -inf;
	for (i = 0; i < in->nTuples; i++)
	{
		Point *temp = DatumGetPointP(in->datums[i]);
		sorted[i].p = temp;
		sorted[i].z_value = compute_Z_value(sorted[i].p, bit_num);
		sorted[i].index = i;
		max_x = Max(temp->x, max_x);
		min_x = Min(temp->x, min_x);
		max_y = Max(temp->y, max_y);
		min_y = Min(temp->y, min_y);
	}

	qsort(sorted, in->nTuples, sizeof(*sorted), z_value_cmp);

	// middle = in->nTuples >> 1;
	// coord = (in->level % 2) ? sorted[middle].p->x : sorted[middle].p->y;
	// if (in->nTuples > 100)
	// {
	// TODO train model
	Model *model = train_model(sorted, in->nTuples);
	int model_num = in->nTuples / 100;

	out->hasPrefix = true;

	model->sub_model_num = (float8)model_num;
	model->cardinality = (float8)in->nTuples;
	model->bit_num = (float8)bit_num;
	BOX *mbr = (BOX *)palloc(sizeof(BOX));
	mbr->high.x = max_x;
	mbr->high.y = max_y;
	mbr->low.x = min_x;
	mbr->low.y = min_y;
	model->mbr = mbr;

	out->prefixDatum = PointerGetDatum(model);

	out->nNodes = model_num;
	out->nodeLabels = NULL; /* we don't need node labels */

	out->mapTuplesToNodes = palloc(sizeof(int) * in->nTuples);
	out->leafTupleDatums = palloc(sizeof(Datum) * in->nTuples);

	for (i = 0; i < in->nTuples; i++)
	{
		out->mapTuplesToNodes[sorted[i].index] = predict(model, sorted[i]);
		out->leafTupleDatums[sorted[i].index] = PointPGetDatum(sorted[i].p);
	}

	PG_RETURN_VOID();
}

Datum
	spg_rsmi_inner_consistent(PG_FUNCTION_ARGS)
{
	// elog(WARNING, "spg_rsmi_inner_consistent");

	spgInnerConsistentIn *in = (spgInnerConsistentIn *)PG_GETARG_POINTER(0);
	spgInnerConsistentOut *out = (spgInnerConsistentOut *)PG_GETARG_POINTER(1);
	double coord;
	int i;
	BOX infbbox;
	BOX *bbox = NULL;
	int which;

	Assert(in->hasPrefix);

	// ArrayType *model_array = DatumGetArrayTypeP(in->prefixDatum);
	// float8 *model_values = (float8 *)ARR_DATA_PTR(model_array);

	Model *model_res = (Model *)DatumGetPointer(in->prefixDatum);
	BOX *mbr = model_res->mbr;

	/*
	 * When ordering scan keys are specified, we've to calculate distance for
	 * them.  In order to do that, we need calculate bounding boxes for all
	 * children nodes.  Calculation of those bounding boxes on non-zero level
	 * require knowledge of bounding box of upper node.  So, we save bounding
	 * boxes to traversalValues.
	 */
	// elog(WARNING, "spg_rsmi_inner_consistent in->norderbys:%d", in->norderbys);

	if (in->norderbys > 0)
	{
		out->distances = (double **)palloc(sizeof(double *) * in->nNodes);
		out->traversalValues = (void **)palloc(sizeof(void *) * in->nNodes);

		if (in->level == 0)
		{
			double inf = get_float8_infinity();

			infbbox.high.x = inf;
			infbbox.high.y = inf;
			infbbox.low.x = -inf;
			infbbox.low.y = -inf;
			bbox = &infbbox;
		}
		else
		{
			bbox = in->traversalValue;
			Assert(bbox);
		}
	}

	elog(WARNING, "spg_rsmi_inner_consistent in->allTheSame:%d in->norderbys:%d", in->allTheSame, in->norderbys);
	if (in->allTheSame) // 1
	{
		/* Report that all nodes should be visited */
		out->nNodes = in->nNodes;
		out->nodeNumbers = (int *)palloc(sizeof(int) * in->nNodes);

		elog(WARNING, "spg_rsmi_inner_consistent in->nNodes:%d", in->nNodes);

		for (i = 0; i < in->nNodes; i++)
		{
			// Point *query = DatumGetPointP(in->scankeys[i].sk_argument);
			// int innerNum = predict_via_array(model_res, query);
			// out->nodeNumbers[i] = innerNum;

			out->nodeNumbers[i] = i;

			if (in->norderbys > 0) // 0
			{
				MemoryContext oldCtx = MemoryContextSwitchTo(in->traversalMemoryContext);

				if (mbr->high.x < bbox->low.x || mbr->high.y < bbox->low.y || mbr->low.x > bbox->high.x || mbr->low.y > bbox->high.y)
					continue;

				BOX *int_mbr = (BOX *)palloc(sizeof(BOX));
				int_mbr->high.x = Min(mbr->high.x, bbox->high.x);
				int_mbr->high.y = Min(mbr->high.y, bbox->high.y);
				int_mbr->low.x = Max(mbr->low.x, bbox->low.x);
				int_mbr->low.y = Max(mbr->low.y, bbox->low.y);

				MemoryContextSwitchTo(oldCtx);
				elog(WARNING, "spg_rsmi_inner_consistent (x1:%f,y1:%f), (x2:%f,y2:%f)", int_mbr->low.x, int_mbr->low.y, int_mbr->high.x, int_mbr->high.y);

				out->traversalValues[i] = int_mbr;
				out->distances[i] = spg_key_orderbys_distances(BoxPGetDatum(int_mbr), false,
															   in->orderbys, in->norderbys);
			}
		}
		PG_RETURN_VOID();
	}

	int left_bound = 0;
	int right_bound = in->nNodes - 1;
	for (i = 0; i < in->nkeys; i++)
	{

		Point *query = DatumGetPointP(in->scankeys[i].sk_argument);
		BOX *boxQuery;
		elog(WARNING, "spg_rsmi_inner_consistent query(x:%f,y:%f)", query->x, query->y);

		int innerNum = predict_via_array(model_res, query);
		elog(WARNING, "spg_rsmi_inner_consistent i:%d ---------- strategy:%d----------- innerNum:%d", i, in->scankeys[i].sk_strategy, innerNum);

		switch (in->scankeys[i].sk_strategy)
		{
		case RTLeftStrategyNumber:
			right_bound = innerNum;
			break;
		case RTRightStrategyNumber:
			left_bound = innerNum;
			break;
		case RTSameStrategyNumber:
			if (innerNum >= left_bound && innerNum <= right_bound)
				which = innerNum;
			// {
			// 	left_bound = innerNum;
			// 	right_bound = innerNum;
			// }
			break;
		case RTBelowStrategyNumber:
			right_bound = innerNum;
			break;
		case RTAboveStrategyNumber:
			left_bound = innerNum;
			break;
		case RTContainedByStrategyNumber:
			boxQuery = DatumGetBoxP(in->scankeys[i].sk_argument);
			int high = predict_via_array(model_res, &boxQuery->high);
			int low = predict_via_array(model_res, &boxQuery->low);

			left_bound = Max(left_bound, low);
			right_bound = Min(right_bound, high);
			break;
		default:
			elog(ERROR, "unrecognized strategy number: %d",
				 in->scankeys[i].sk_strategy);
			break;
		}

		if (right_bound < left_bound)
			break; /* no need to consider remaining conditions */
	}
	elog(WARNING, "spg_rsmi_inner_consistent left_bound:%d ---------- right_bound:%d  in->nNodes:%d", left_bound, right_bound, in->nNodes);

	out->levelAdds = palloc(sizeof(int) * in->nNodes);
	for (i = 0; i < in->nNodes; ++i)
		out->levelAdds[i] = 1;

	out->nodeNumbers = (int *)palloc(sizeof(int) * (right_bound - left_bound + 1));
	out->nNodes = 0;

	for (i = left_bound; i <= right_bound; i++)
	{
		out->nodeNumbers[out->nNodes] = i;

		if (in->norderbys > 0)
		{
			MemoryContext oldCtx = MemoryContextSwitchTo(in->traversalMemoryContext);

			if (mbr->high.x < bbox->low.x || mbr->high.y < bbox->low.y || mbr->low.x > bbox->high.x || mbr->low.y > bbox->high.y)
				continue;

			BOX *int_mbr = (BOX *)palloc(sizeof(BOX));
			int_mbr->high.x = Min(mbr->high.x, bbox->high.x);
			int_mbr->high.y = Min(mbr->high.y, bbox->high.y);
			int_mbr->low.x = Max(mbr->low.x, bbox->low.x);
			int_mbr->low.y = Max(mbr->low.y, bbox->low.y);
			MemoryContextSwitchTo(oldCtx);

			out->traversalValues[out->nNodes] = int_mbr;

			out->distances[out->nNodes] = spg_key_orderbys_distances(BoxPGetDatum(int_mbr), false,
																	 in->orderbys, in->norderbys);
		}

		out->nNodes++;
	}
	PG_RETURN_VOID();
}

/*
 * spg_rsmi_leaf_consistent() is the same as spg_quad_leaf_consistent(),
 * since we support the same operators and the same leaf data type.
 * So we just borrow that function.
 */
