# Generated from PS.g4 by ANTLR 4.9.3
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .PSParser import PSParser
else:
    from PSParser import PSParser

# This class defines a complete generic visitor for a parse tree produced by PSParser.

class PSVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by PSParser#accent_symbol.
    def visitAccent_symbol(self, ctx:PSParser.Accent_symbolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#math.
    def visitMath(self, ctx:PSParser.MathContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#transpose.
    def visitTranspose(self, ctx:PSParser.TransposeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#transform_atom.
    def visitTransform_atom(self, ctx:PSParser.Transform_atomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#transform_scale.
    def visitTransform_scale(self, ctx:PSParser.Transform_scaleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#transform_swap.
    def visitTransform_swap(self, ctx:PSParser.Transform_swapContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#transform_assignment.
    def visitTransform_assignment(self, ctx:PSParser.Transform_assignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#elementary_transform.
    def visitElementary_transform(self, ctx:PSParser.Elementary_transformContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#elementary_transforms.
    def visitElementary_transforms(self, ctx:PSParser.Elementary_transformsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#matrix.
    def visitMatrix(self, ctx:PSParser.MatrixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#det.
    def visitDet(self, ctx:PSParser.DetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#matrix_row.
    def visitMatrix_row(self, ctx:PSParser.Matrix_rowContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#relation.
    def visitRelation(self, ctx:PSParser.RelationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#relation_list.
    def visitRelation_list(self, ctx:PSParser.Relation_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#relation_list_content.
    def visitRelation_list_content(self, ctx:PSParser.Relation_list_contentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#equality.
    def visitEquality(self, ctx:PSParser.EqualityContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#expr.
    def visitExpr(self, ctx:PSParser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#additive.
    def visitAdditive(self, ctx:PSParser.AdditiveContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#mp.
    def visitMp(self, ctx:PSParser.MpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#mp_nofunc.
    def visitMp_nofunc(self, ctx:PSParser.Mp_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#unary.
    def visitUnary(self, ctx:PSParser.UnaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#unary_nofunc.
    def visitUnary_nofunc(self, ctx:PSParser.Unary_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#postfix.
    def visitPostfix(self, ctx:PSParser.PostfixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#postfix_nofunc.
    def visitPostfix_nofunc(self, ctx:PSParser.Postfix_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#postfix_op.
    def visitPostfix_op(self, ctx:PSParser.Postfix_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#eval_at.
    def visitEval_at(self, ctx:PSParser.Eval_atContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#eval_at_sub.
    def visitEval_at_sub(self, ctx:PSParser.Eval_at_subContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#eval_at_sup.
    def visitEval_at_sup(self, ctx:PSParser.Eval_at_supContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#exp.
    def visitExp(self, ctx:PSParser.ExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#exp_nofunc.
    def visitExp_nofunc(self, ctx:PSParser.Exp_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#comp.
    def visitComp(self, ctx:PSParser.CompContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#comp_nofunc.
    def visitComp_nofunc(self, ctx:PSParser.Comp_nofuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#group.
    def visitGroup(self, ctx:PSParser.GroupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#norm_group.
    def visitNorm_group(self, ctx:PSParser.Norm_groupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#abs_group.
    def visitAbs_group(self, ctx:PSParser.Abs_groupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#floor_group.
    def visitFloor_group(self, ctx:PSParser.Floor_groupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#ceil_group.
    def visitCeil_group(self, ctx:PSParser.Ceil_groupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#accent.
    def visitAccent(self, ctx:PSParser.AccentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#atom_expr_no_supexpr.
    def visitAtom_expr_no_supexpr(self, ctx:PSParser.Atom_expr_no_supexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#atom_expr.
    def visitAtom_expr(self, ctx:PSParser.Atom_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#atom.
    def visitAtom(self, ctx:PSParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#mathit.
    def visitMathit(self, ctx:PSParser.MathitContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#mathit_text.
    def visitMathit_text(self, ctx:PSParser.Mathit_textContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#frac.
    def visitFrac(self, ctx:PSParser.FracContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#binom.
    def visitBinom(self, ctx:PSParser.BinomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_normal_functions_single_arg.
    def visitFunc_normal_functions_single_arg(self, ctx:PSParser.Func_normal_functions_single_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_normal_functions_multi_arg.
    def visitFunc_normal_functions_multi_arg(self, ctx:PSParser.Func_normal_functions_multi_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_operator_names_single_arg.
    def visitFunc_operator_names_single_arg(self, ctx:PSParser.Func_operator_names_single_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_operator_names_multi_arg.
    def visitFunc_operator_names_multi_arg(self, ctx:PSParser.Func_operator_names_multi_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_normal_single_arg.
    def visitFunc_normal_single_arg(self, ctx:PSParser.Func_normal_single_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_normal_multi_arg.
    def visitFunc_normal_multi_arg(self, ctx:PSParser.Func_normal_multi_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func.
    def visitFunc(self, ctx:PSParser.FuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#args.
    def visitArgs(self, ctx:PSParser.ArgsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_common_args.
    def visitFunc_common_args(self, ctx:PSParser.Func_common_argsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#limit_sub.
    def visitLimit_sub(self, ctx:PSParser.Limit_subContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_single_arg.
    def visitFunc_single_arg(self, ctx:PSParser.Func_single_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_single_arg_noparens.
    def visitFunc_single_arg_noparens(self, ctx:PSParser.Func_single_arg_noparensContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_multi_arg.
    def visitFunc_multi_arg(self, ctx:PSParser.Func_multi_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#func_multi_arg_noparens.
    def visitFunc_multi_arg_noparens(self, ctx:PSParser.Func_multi_arg_noparensContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#subexpr.
    def visitSubexpr(self, ctx:PSParser.SubexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#supexpr.
    def visitSupexpr(self, ctx:PSParser.SupexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#subeq.
    def visitSubeq(self, ctx:PSParser.SubeqContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by PSParser#supeq.
    def visitSupeq(self, ctx:PSParser.SupeqContext):
        return self.visitChildren(ctx)



del PSParser