// Trait PointsToInfo
//trait PointsToInfo {
//    PointsToSet get_points_to_set(pointer: P, instruction: I);
//}

// Trait AliasInfo
//trait AliasInfo {
//    AliasSet get_alias_set(pointer: P, instruction: I);
//}

use std::collections::HashMap;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct FunctionIndex(u32);

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct StatementIndex(u32);

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct ProgramPos {
    function_index: FunctionIndex,
    statement_index: StatementIndex,
}

struct ICFGEdge {
    current: ProgramPos,
    successor: ProgramPos,
}

// TODO: Summaries!
trait ICFG {
    fn get_successors(&self, pos: ProgramPos) -> Vec<ProgramPos>; // TODO: What about return statement? How are they able to return the successor function_index without knowing the call stack?
    fn get_call_instructions_in_function_as_program_pos(&self, function: FunctionIndex) -> Vec<ProgramPos>;
    fn all_instructions(&self) -> Vec<ProgramPos>;
    fn is_starting_point_or_call_site(&self, pos: ProgramPos) -> bool;
    fn get_starting_point_for(&self, function: FunctionIndex) -> ProgramPos;
}

// Open Question: Can we really "remove" the EdgeFunction concept? How to represent it instead?

trait Joinable {
    fn join_with(&self, edge2: Self) -> Self;
}
trait Composable {
    fn compose_with(&self, edge2: Self) -> Self; // sequential application of two edge functions (first x+=1, then x+=2 -> results in x+=3)
}
trait Computable<ConcreteValue> {
    fn compute(&self, input: &ConcreteValue) -> ConcreteValue; // evaluates this edge function for a concrete value, returning a resulting value of the same type.
}


pub mod ide {
    use std::collections::{HashMap, HashSet};
    use std::hash::Hash;
    use super::*;

    struct IDEJumpFunctionTable<P: IDEProblem> {
        x: Vec<HashMap<JumpFunctionKey<P>, P::Weight>>
    }
    impl <P: IDEProblem> Default for IDEJumpFunctionTable<P> {
        fn default() -> Self {
            IDEJumpFunctionTable { x: vec![] }
        }
    }
    impl <P: IDEProblem> IDEJumpFunctionTable<P> {
        pub fn already_visited(&self, item: &Phase1WorklistItem<P>) -> bool { todo!() }
        pub fn get_at_program_pos(&self, pos: ProgramPos) -> P::Weight {
            todo!()
        }

    }
    trait IDEProblem {
        type FlowFact: Clone + Hash + Eq;
        type ConcreteValue: Clone + Joinable + PartialEq;
        type Weight: Clone + Composable + Computable<Self::ConcreteValue> + Joinable + PartialEq;

        fn apply_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
        //fn apply_normal_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
        //fn apply_call_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
        //fn apply_return_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
        //fn apply_call_to_return_flow(icfg_edge: ICFGEdge, data_flow_fact: Self::FlowFact) -> Vec<Self::FlowFact>;
    }

    struct EntryPoint<P: IDEProblem> {
        instruction: ProgramPos,
        flow_fact: P::FlowFact,
        concrete_value: P::ConcreteValue,
    }

    struct SolverResults<P: IDEProblem> {
        concrete_values: HashMap<(ProgramPos, P::FlowFact), P::ConcreteValue>,
    }
    impl <P: IDEProblem> Default for SolverResults<P> {
        fn default() -> Self {
            SolverResults { concrete_values: HashMap::default() }
        }
    }
    impl <P: IDEProblem> SolverResults<P> {
        fn insert_or_join(&mut self, weight: P::Weight, flow_fact: P::FlowFact, computed_value: P::ConcreteValue) -> bool {
            todo!()
        }
    }


    #[derive(Hash)]
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
        edge_function: P::Weight, // ?
    }
    impl <P: IDEProblem> Phase1WorklistItem<P> {
        pub fn new_with_entry_point(entry_point: &EntryPoint<P>) -> Self {
            todo!()
        }
        fn get_successor_worklist_items(&self) -> Vec<Self> {
            todo!()
        }
    }

    struct Phase2WorklistItem<P: IDEProblem> {
        instruction: ProgramPos,
        flow_fact: P::FlowFact,
        concrete_value: P::ConcreteValue,
    }
    fn propagate_into_callees<P: IDEProblem>(call_instruction: ProgramPos) -> Vec<Phase2WorklistItem<P>> {
        todo!()
    }


    fn solve<P: IDEProblem>(problem: P, entry_points: Vec<EntryPoint<P>>, icfg: &impl ICFG) -> SolverResults<P> {
    
        // Jump Function = Edge in CFG.
        // Starting point: First instruction of current function with the flow facts that hold before that instructions (passed in by call site).
        // End point (second part of key in hashmap): Last inst
    
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
                worklist.push(Phase1WorklistItem::new_with_entry_point(entry_point));
            }
            while let Some(current) = worklist.pop() {
                for successor in current.get_successor_worklist_items() {
                    if !ide_jump_function_table.already_visited(&successor) {
                        worklist.push(successor);
                    }
                }
            }
        }

        {
            // Phase 2.1: Compute concrete values by propagating at calls into callees leading us concrete values at all call instructions and function starting points naturally.
            // TODO: This happens to hold the same data as the EntryPoint struct! Unify?
            let mut worklist: Vec<Phase2WorklistItem<P>> = vec![];
            let mut results: SolverResults<P> = SolverResults::default();

            for entry_point in &entry_points {
                worklist.push(Phase2WorklistItem {
                    instruction: entry_point.instruction,
                    flow_fact: entry_point.flow_fact.clone(),
                    concrete_value: entry_point.concrete_value.clone()
                });
            }

            while let Some(item) = worklist.pop() {
                let function = item.instruction.function_index;
                for call_instr in icfg.get_call_instructions_in_function_as_program_pos(function) {
                    let weight: P::Weight = ide_jump_function_table.get_at_program_pos(call_instr);
                    let computed_value = weight.compute(&item.concrete_value);
                    let flow_fact = todo!(); // TODO
                    let was_updated = results.insert_or_join(weight, flow_fact, computed_value);
                    if was_updated {
                        worklist.append(&mut propagate_into_callees(call_instr));
                    }
                }
            }



            // Phase 2.2: Compute all concrete values for all instructions not yet visited in Phase 2.1
            for program_pos in icfg.all_instructions() {
                if icfg.is_starting_point_or_call_site(program_pos) {
                    // Note: These are already treated in Phase 2.1
                    continue;
                }
                //let starting_point = icfg.get_starting_point_for(program_pos.function_index);
                let starting_point_fact = todo!(); // How do we get the starting point fact?
                let weight: P::Weight = ide_jump_function_table.get_at_program_pos(program_pos);

                if let Some(value) = results.concrete_values.get(&(program_pos, starting_point_fact)) {
                    let computed_value = weight.compute(value);
                    let flow_fact = todo!(); // TODO
                    results.concrete_values.insert((program_pos, flow_fact), computed_value);
                }
            }

            return results;
        }
    }
}

fn main() {

}

#[cfg(tests)]
mod tests {
    #[test]
    fn test_dummy() {

    }
}