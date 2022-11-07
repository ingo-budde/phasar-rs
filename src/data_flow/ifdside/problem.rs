use crate::{
    control_flow::{ICFGEdge, ProgramPos},
    data_flow::JoinLattice,
    ir::IRDescription,
};
use std::hash::Hash;

use super::IDEWeight;

pub trait AnalysisDomain {
    type IR: IRDescription;
    type FlowFact: Copy + Hash + Eq;
    type ConcreteValue: JoinLattice;
}

pub struct ResultEntry<Domain: AnalysisDomain> {
    instruction: ProgramPos<Domain::IR>,
    flow_fact: Domain::FlowFact,
    concrete_value: Domain::ConcreteValue,
}

pub trait IDEProblem<Domain: AnalysisDomain> {
    type Weight: IDEWeight<Domain::ConcreteValue>;

    // TODO: again, wish to return an impl Iterator instead
    fn apply_flow(
        &self,
        edge: ICFGEdge<Domain::IR>,
        source: Domain::FlowFact,
    ) -> Vec<(Domain::FlowFact, Self::Weight)>;

    fn get_initial_seeds(
        &self,
        entry_functions: impl Iterator<Item = <Domain::IR as IRDescription>::Function>,
    ) -> Vec<ResultEntry<Domain>>;

    fn is_all_identity(&self, pos: ProgramPos<Domain::IR>) -> bool;
}

pub trait AnalysisResults<Domain: AnalysisDomain> {
    fn has_results_at(&self, at_inst: <Domain::IR as IRDescription>::Instruction) -> bool;
    fn holds_fact(
        &self,
        at_inst: <Domain::IR as IRDescription>::Instruction,
        fact: Domain::FlowFact,
    ) -> bool;
    fn get_value_or_top(
        &self,
        at_inst: <Domain::IR as IRDescription>::Instruction,
        fact: Domain::FlowFact,
    ) -> Domain::ConcreteValue;

    // fn iter(&self) -> impl Iterator<Item = ResultEntry<Domain>>;
}
