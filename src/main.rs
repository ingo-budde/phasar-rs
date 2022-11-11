mod cli;
mod utils;


use crate::cli::CommandLineArgs;

use crate::icfg::{FunctionIndex, ProgramPos, StatementIndex};
use crate::ide::EntryPoint;

mod aliases {
    // Trait PointsToInfo
    //trait PointsToInfo {
    //    PointsToSet get_points_to_set(pointer: P, instruction: I);
    //}

    // Trait AliasInfo
    //trait AliasInfo {
    //    AliasSet get_alias_set(pointer: P, instruction: I);
    //}
}

mod icfg {
    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    pub struct FunctionIndex(pub u32);
    impl FunctionIndex { pub fn new(index: usize) -> FunctionIndex { FunctionIndex(index as u32) } }

    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    pub struct StatementIndex(pub u32);
    impl StatementIndex { pub fn new(index: usize) -> StatementIndex { StatementIndex(index as u32) } }

    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    pub struct ProgramPos {
        pub function: FunctionIndex,
        pub statement: StatementIndex,
    }
    impl ProgramPos {
        pub fn new(function: FunctionIndex, statement: StatementIndex) -> Self {
            ProgramPos { function, statement }
        }
        pub fn function_start(function: FunctionIndex) -> ProgramPos {
            ProgramPos { function, statement: StatementIndex(0) }
        }
        pub fn try_next_position(&self, num_instructions: usize) -> Option<ProgramPos> {
            if let Some(next_statement) = self.statement.0.checked_add(1) {
                if (next_statement as usize) < num_instructions {
                    return Some(ProgramPos {
                        function: self.function,
                        statement: StatementIndex(next_statement)
                    })
                }
            }
            None
        }
    }

    pub struct ICFGEdge {
        pub current: ProgramPos,
        pub successor: ProgramPos,
    }

    // TODO: Summaries!
    pub trait ICFG {
        fn get_successors(&self, pos: ProgramPos) -> Vec<ProgramPos>; // TODO: What about return statement? How are they able to return the successor function_index without knowing the call stack?
        fn get_call_instructions_in_function_as_program_pos(&self, function: FunctionIndex) -> Vec<ProgramPos>;
        fn all_instructions(&self) -> Vec<ProgramPos>;
        fn is_starting_point_or_call_site(&self, pos: ProgramPos) -> bool;
        fn get_starting_point_for(&self, function: FunctionIndex) -> ProgramPos;
    }
}

pub trait Joinable {
    fn join_with(&self, edge2: &Self) -> Self; // merge multiple branches potentially loosing information.
}
pub trait Composable {
    fn compose_with(&self, edge2: &Self) -> Self; // sequential application of two edge functions (first x+=1, then x+=2 -> results in x+=3)
}
pub trait Computable<ConcreteValue> {
    fn compute(&self, input: &ConcreteValue) -> ConcreteValue; // evaluates this edge function for a concrete value, returning a resulting value of the same type.
}


pub mod ide {
    use std::collections::{HashMap};
    use std::fmt::{Debug, Formatter};
    use std::hash::{Hash};
    use super::{*, icfg::*};

    #[derive(Debug)]
    struct IDEJumpFunctionTable<P: IDEProblem> {
        // TODO: Replace the outer HashMap by a vector? -> Come up with a good key...
        data: HashMap<ProgramPos, HashMap<JumpFunctionKey<P>, P::Weight>>,
    }
    impl <P: IDEProblem> Default for IDEJumpFunctionTable<P> {
        fn default() -> Self {
            IDEJumpFunctionTable { data: Default::default() }
        }
    }
    impl <P: IDEProblem> IDEJumpFunctionTable<P> {
        pub fn handle_worklist_item(&mut self, item: &Phase1WorklistItem<P>) -> bool {
            let x = self.data.entry(item.pos).or_default();
            let key = JumpFunctionKey { fact_at_start: item.source_fact.clone(), fact_at_end: item.propagated_fact.clone() };
            if let Some(weight) = x.get_mut(&key) {
                let new_weight = weight.join_with(&item.weight);
                if new_weight == *weight {
                    return false;
                } else {
                    *weight = new_weight;
                    return true;
                }
            } else {
                x.insert(key, item.weight.clone());
                return true;
            }
        }
        pub fn get_at_program_pos(&self, pos: ProgramPos, key: JumpFunctionKey<P> ) -> Option<&P::Weight> {
            self.data.get(&pos)?.get(&key)
        }
    }
    pub trait IDEProblem: Hash + Eq + Debug + PartialEq { // Note: We require "IDEProblem: Hash + Eq + PartialEq", just to be able to derive those traits at structs using "P: FlowFact where P: IDEProblem" - see rust bug https://github.com/rust-lang/rust/issues/26925
        type FlowFact: Copy + Clone + Hash + Eq + Debug + PartialEq;
        type ConcreteValue: Copy + Clone + Joinable + PartialEq + Debug + Default;
        type Weight: Copy + Clone + Composable + Computable<Self::ConcreteValue> + Joinable + PartialEq + Debug;
        type ControlFlowGraph: ICFG;

        fn get_program_pos(&self, flow_fact: Self::FlowFact) -> ProgramPos;
        fn get_flow_fact(&self, program_pos: ProgramPos) -> Self::FlowFact;

        fn apply_flow(&self, icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
        //fn apply_normal_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
        //fn apply_call_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
        //fn apply_return_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
        //fn apply_call_to_return_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;

        fn get_control_flow_graph(&self) -> &Self::ControlFlowGraph;
    }

    pub struct EntryPoint<P: IDEProblem> {
        pub instruction: ProgramPos,
        pub flow_fact: P::FlowFact,
        pub concrete_value: P::ConcreteValue,
        pub weight: P::Weight,
    }

    pub struct SolverResults<P: IDEProblem> {
        pub concrete_values: HashMap<(ProgramPos, P::FlowFact), P::ConcreteValue>,
    }
    impl <P: IDEProblem> Debug for SolverResults<P> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            self.concrete_values.fmt(f)
        }
    }
    impl <P: IDEProblem> Default for SolverResults<P> {
        fn default() -> Self {
            SolverResults { concrete_values: HashMap::default() }
        }
    }
    impl <P: IDEProblem> SolverResults<P> {
        fn insert_or_join(&mut self, _weight: P::Weight, _flow_fact: P::FlowFact, _computed_value: P::ConcreteValue) -> bool {
            todo!()
        }
    }

    // Jump Function Key = Edge in CFG.
    // Starting point: First instruction of current function with the flow facts that hold before that instructions (passed in by call site).
    // End point (second part of key in hashmap): Last instruction of the Jump Function (i.e. up to the point that the jump function has been built), ultimately the exit point of the function
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct JumpFunctionKey<P: IDEProblem> {
        fact_at_start: P::FlowFact,

        // Note: We incrementally extend the jump-function-table,
        // so while building it up, this is actually the fact until which we built the jump function.
        // Note2: This also represents the tainted variable.
        fact_at_end: P::FlowFact,
    }




    struct Phase1WorklistItem<P: IDEProblem> {
        // IFDS specific:
        pos: ProgramPos,
        source_fact: P::FlowFact,
        propagated_fact: P::FlowFact,

        // IDE specific:
        weight: P::Weight, // ?
    }
    impl <P: IDEProblem> Phase1WorklistItem<P> {
        pub fn new_with_entry_point(entry_point: &EntryPoint<P>) -> Self {
            Self {
                pos: entry_point.instruction,
                source_fact: entry_point.flow_fact.clone(),
                propagated_fact: entry_point.flow_fact.clone(), // TODO not really
                weight: entry_point.weight.clone()
            }
        }
        fn get_successor_worklist_items(&self, problem: &P) -> Vec<Self> {
            // TODO: This calculates the cross product: get_successors() X apply_flow()

            let new_source_fact = self.propagated_fact;
            let new_current_pos = problem.get_program_pos(new_source_fact);

            problem.get_control_flow_graph().get_successors(new_current_pos)
                .into_iter()
                .flat_map(|new_successor_pos| {
                    let icfg_edge = ICFGEdge {
                        current: new_current_pos,
                        successor: new_successor_pos,
                    };
                    problem.apply_flow(icfg_edge, self.propagated_fact)
                        .into_iter()
                        .map(|propagated_fact| {
                            Phase1WorklistItem {
                                pos: new_current_pos,
                                source_fact: new_source_fact,
                                propagated_fact,
                                weight: self.weight.clone() // TODO
                            }
                        })
                })
                .collect()
        }
    }

    struct Phase2WorklistItem<P: IDEProblem> {
        instruction: ProgramPos,
        concrete_value: P::ConcreteValue,
    }

    fn propagate_into_callees<P: IDEProblem>(_call_instruction: ProgramPos) -> Vec<Phase2WorklistItem<P>> {
        todo!()
    }

    pub fn solve<P: IDEProblem>(problem: P, entry_points: Vec<EntryPoint<P>>) -> SolverResults<P> {

        // FlowFact = Variable für die wir die Jump Function berechnen, oder allgemein: Was wir mit der Analyse "verfolgen"; wofür wir die EdgeValues berechnen.
        // The first flow fact: 
        // The second flow fact is the tainted variable.
        // 

        // IDE: We have weights. Jump function is a HashMap.
        // IFDS: We don't have weights. Jump function is a HashSet which just tells us that we visited this worklist item already.
        
        // We can remove items for finished functions with completely built JumpFunctions again
        // let mut already_visited: HashSet<WorklistItem> = HashSet::new();
        let mut ide_jump_function_table = IDEJumpFunctionTable::default();
        //let mut ifds_jump_function_set: Vec<HashSet<JumpFunctionKey>> = vec![];

        // Phase 1 (Build-up Exploded Supergraph (ESG) (=jump functions) - until we reach fixpoint)
        {
            let mut worklist: Vec<Phase1WorklistItem<P>> = vec![];
            for entry_point in &entry_points {
                let item = Phase1WorklistItem::new_with_entry_point(entry_point);
                if ide_jump_function_table.handle_worklist_item(&item) {
                    worklist.push(item);
                }
            }
            while let Some(current) = worklist.pop() {
                for successor in current.get_successor_worklist_items(&problem) {
                    if ide_jump_function_table.handle_worklist_item(&successor) {
                        worklist.push(successor);
                    }
                }
            }
        }

        log::info!("Calculated Exploded Supergraph (ESG): {:#?}", ide_jump_function_table);

        {
            // Phase 2.1: Compute concrete values by propagating at calls into callees leading us concrete values at all call instructions and function starting points naturally.
            let mut worklist: Vec<Phase2WorklistItem<P>> = vec![];
            let mut results: SolverResults<P> = SolverResults::default();

            for entry_point in &entry_points {
                worklist.push(Phase2WorklistItem {
                    instruction: entry_point.instruction,
                    concrete_value: entry_point.concrete_value.clone()
                });
            }
            let icfg = problem.get_control_flow_graph();
            while let Some(item) = worklist.pop() {
                let function = item.instruction.function;
                let starting_point = icfg.get_starting_point_for(function);
                for call_instr in icfg.get_call_instructions_in_function_as_program_pos(function) {
                    let starting_point_fact = problem.get_flow_fact(starting_point);
                    let call_site_fact = problem.get_flow_fact(call_instr);
                    let key: JumpFunctionKey<P> = JumpFunctionKey { fact_at_start: starting_point_fact, fact_at_end: call_site_fact.clone() };
                    if let Some(weight) = ide_jump_function_table.get_at_program_pos(call_instr, key) {
                        let computed_value = weight.compute(&item.concrete_value);
                        let was_updated = results.insert_or_join(weight.clone(), call_site_fact, computed_value);
                        if was_updated {
                            worklist.append(&mut propagate_into_callees(call_instr));
                        }
                    }
                }
            }



            // Phase 2.2
            // Compute all concrete values for all instructions not yet visited in Phase 2.1
            // (all instructions except call instructions and function starting points)
            for program_pos in icfg.all_instructions() {
                if icfg.is_starting_point_or_call_site(program_pos) {
                    // Note: These are already treated in Phase 2.1
                    continue;
                }
                let starting_point = icfg.get_starting_point_for(program_pos.function);
                let starting_point_fact = problem.get_flow_fact(starting_point);
                let current_fact = problem.get_flow_fact(program_pos);
                let key: JumpFunctionKey<P> = JumpFunctionKey { fact_at_start: starting_point_fact.clone(), fact_at_end: current_fact };
                if let Some(weight) = ide_jump_function_table.get_at_program_pos(program_pos, key) {
                    if let Some(value) = results.concrete_values.get(&(program_pos, starting_point_fact)) {
                        let computed_value = weight.compute(value);
                        let flow_fact = problem.get_flow_fact(program_pos);
                        results.concrete_values.insert((program_pos, flow_fact), computed_value);
                    }
                }

            }

            return results;
        }
    }
}


/// Here, we define a toy-like example IR for describing taint flow specific problems.
///
/// In Phasar, the solver is written in a generic way and does not assume a specific IR format.
/// This makes it possible to support different IR's, which can be hand-crafted for specific problems.
///
/// The advantage of a problem-specific IR is that it usually leads to a smaller Instruction-Set,
/// which simplifies the overall implementation. But this also means that we need to transform the
/// actual program into the problem-specific IR ahead-of-time, dropping all information that is
/// not relevant for the problem domain.
///
/// In a real-world scenario, we would additionally need to have some way to map any IR statement back
/// to the original statement, or at least it's line and column number, so we are able to report findings
/// at the right places in the actual source code.
mod example_taint_flow_ir {
    use std::fmt::{Debug};
    use std::hash::{Hash, Hasher};
    use crate::icfg::{FunctionIndex, ICFG, ICFGEdge, ProgramPos};
    use crate::ide::{IDEProblem};
    use crate::{Composable, Computable, Joinable, StatementIndex};

    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    pub struct LocalVarId(pub u32);

    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    pub enum VariableId {
        Local(LocalVarId)
    }
    #[derive(Debug)]
    pub enum Statement {
        Source(SourceStmt),
        Assign(AssignStmt),
        Sink(SinkStmt),
        SetConstant(SetConstantStmt),
        Call(CallStmt),
        Return(ReturnStmt),
    }
    impl Statement {
        pub fn is_call(&self) -> bool { if let Statement::Call(_) = self { true } else { false } }
    }
    #[derive(Debug)]
    pub struct SourceStmt {
        pub tainting_var: VariableId,
    }
    #[derive(Debug)]
    pub struct AssignStmt {
        pub lhs: VariableId,
        pub rhs: VariableId,
    }
    #[derive(Debug)]
    pub struct SinkStmt {
        pub relevant_var: VariableId
    }
    #[derive(Debug)]
    pub struct SetConstantStmt {
        pub lhs: VariableId,
        pub rhs: DummyConcreteValue,
    }
    #[derive(Debug)]
    pub struct CallStmt {
        pub return_var: Option<VariableId>,
        pub arguments: Vec<Option<VariableId>>,
        pub function: FunctionIndex,
    }
    #[derive(Debug)]
    pub struct ReturnStmt {
        pub return_value: Option<VariableId>,
    }
    #[derive(Debug)]
    pub struct Program {
        pub functions: Vec<Function>,
    }
    impl Program {
        fn get_stmt(&self, pos: ProgramPos) -> &Statement {
            let f = &self.functions[pos.function.0 as usize];
            &f.statements[pos.statement.0 as usize]
        }
    }
    impl ICFG for Program {
        fn get_successors(&self, pos: ProgramPos) -> Vec<ProgramPos> {
            let f = &self.functions[pos.function.0 as usize];
            match &f.statements[pos.statement.0 as usize] {
                Statement::Source(_) | Statement::Assign(_) | Statement::Sink(_) | Statement::SetConstant(_)  => {
                    if let Some(next) = pos.try_next_position(f.statements.len()) {
                        vec![next]
                    } else {
                        vec![]
                    }
                }
                Statement::Call(call) => {
                    vec![ProgramPos::function_start(call.function)]
                }
                Statement::Return(_) => {
                    todo!() // TODO: How to implement this?
                }
            }
        }

        fn get_call_instructions_in_function_as_program_pos(&self, function: FunctionIndex) -> Vec<ProgramPos> {
            let f = &self.functions[function.0 as usize];
            f.statements
                .iter()
                .enumerate()
                .filter(|(_, stmt)| stmt.is_call())
                .map(|x| ProgramPos::new(function, StatementIndex(x.0 as u32)))
                .collect()
        }

        fn all_instructions(&self) -> Vec<ProgramPos> {
            self.functions
                .iter()
                .enumerate()
                .flat_map(|(function_index, f) |
                    (0..f.statements.len())
                        .map(move |statement_index| ProgramPos::new(
                            FunctionIndex::new(function_index),
                            StatementIndex::new(statement_index)
                        ))
                )
                .collect()
        }

        fn is_starting_point_or_call_site(&self, pos: ProgramPos) -> bool {
            pos.statement.0 == 0 || self.get_stmt(pos).is_call()
        }

        fn get_starting_point_for(&self, function: FunctionIndex) -> ProgramPos {
            ProgramPos::function_start(function)
        }
    }
    #[derive(Debug)]
    pub struct Function {
        pub statements: Vec<Statement>
    }
    #[derive(Debug)]
    pub struct TaintFlowProblem {
        pub program: Program,
    }
    // Note: We require "IDEProblem: Hash + Eq + PartialEq", just to be able to derive those traits at structs using "P: FlowFact where P: IDEProblem" - see rust bug https://github.com/rust-lang/rust/issues/26925
    impl PartialEq for TaintFlowProblem { fn eq(&self, _other: &Self) -> bool { panic!("Tried to compare TaintFlowProblem") } }
    impl Eq for TaintFlowProblem { }
    impl Hash for TaintFlowProblem { fn hash<H: Hasher>(&self, _state: &mut H) {  panic!("Tried to hash TaintFlowProblem") } }

    #[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Default)]
    pub struct DummyConcreteValue;

    #[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
    pub enum DummyWeight {
        NoWeight, // i.e. "Bottom"
        //...
    }
    impl Default for DummyWeight {
        fn default() -> Self { DummyWeight::NoWeight }
    }

    impl Joinable for DummyConcreteValue { fn join_with(&self, _edge2: &Self) -> Self { todo!() } }
    impl Joinable for DummyWeight { fn join_with(&self, _edge2: &Self) -> Self { todo!() } }
    impl Computable<DummyConcreteValue> for DummyWeight { fn compute(&self, _input: &DummyConcreteValue) -> DummyConcreteValue { todo!() } }
    impl Composable for DummyWeight { fn compose_with(&self, _edge2: &Self) -> Self { todo!() } }


    impl IDEProblem for TaintFlowProblem {
        type FlowFact = ProgramPos;
        type ConcreteValue = DummyConcreteValue;
        type Weight = DummyWeight;
        type ControlFlowGraph = Program;

        fn get_program_pos(&self, flow_fact: Self::FlowFact) -> ProgramPos { flow_fact }
        fn get_flow_fact(&self, program_pos: ProgramPos) -> Self::FlowFact {
            program_pos
        }

        fn apply_flow(&self, icfg_edge: ICFGEdge, _data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact> {
            vec![self.get_flow_fact(icfg_edge.successor)]
        }

        fn get_control_flow_graph(&self) -> &Self::ControlFlowGraph { &self.program }
    }
}


fn main() {
    // Parse CLI arguments (e.g. -v, -vv, -vvv) to activate verbose logging or quiet mode with -q.
    use clap::Parser;
    let cli_arguments = CommandLineArgs::parse();
    cli_arguments.init_logger();

    use example_taint_flow_ir::*;
    let problem = TaintFlowProblem {
        program: Program {
            functions: vec![
                Function {
                    statements: vec![
                        // $0 = source()
                        Statement::Source(SourceStmt { tainting_var: VariableId::Local(LocalVarId(0))}),

                        // $1 = $0
                        Statement::Assign(AssignStmt {
                            lhs: VariableId::Local(LocalVarId(1)),
                            rhs: VariableId::Local(LocalVarId(0)),
                        }),

                        // sink($1)
                        Statement::Sink(SinkStmt { relevant_var: VariableId::Local(LocalVarId(1)) }),
                    ]
                }
            ]
        }
    };

    let entry_points = vec![
        EntryPoint {
            concrete_value: DummyConcreteValue,
            instruction: ProgramPos::new(FunctionIndex(0), StatementIndex(0)),
            flow_fact: ProgramPos::new(FunctionIndex(0), StatementIndex(0)),
            weight: DummyWeight::NoWeight
        }
    ];

    let results = ide::solve(problem, entry_points);

    log::info!("Got results: {:#?}", results);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ir() {

    }
}