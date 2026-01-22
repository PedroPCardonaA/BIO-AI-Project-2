use std::collections::HashSet;
use crate::structs::instance::Instance;

/// Represents the result of a solution validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_feasible: bool,
    pub violations: Vec<String>,
}

impl ValidationResult {
    pub fn new() -> Self {
        ValidationResult {
            is_feasible: true,
            violations: Vec::new(),
        }
    }
    
    pub fn add_violation(&mut self, violation: String) {
        self.is_feasible = false;
        self.violations.push(violation);
    }
}

/// Validates if a solution is feasible according to all constraints
/// 
/// # Constraints checked:
/// 1. Each route starts at the depot on time 0
/// 2. Each route ends at the depot and must arrive before the depot return time
/// 3. The total demand on a route must be <= nurse's capacity
/// 4. Each patient visit on a route must be within the respective time windows
/// 5. Each patient is visited exactly once
/// 
/// # Parameters
/// - `solution`: The solution to validate (vector of routes)
/// - `instance`: The problem instance with all data
/// 
/// # Returns
/// A `ValidationResult` containing feasibility status and list of violations
pub fn validate_solution(solution: &Vec<Vec<usize>>, instance: &Instance) -> ValidationResult {
    let mut result = ValidationResult::new();
    
    // Constraint 5: Check that each patient is visited exactly once
    check_patient_coverage(solution, instance, &mut result);
    
    // Check constraints for each route
    for (nurse_idx, route) in solution.iter().enumerate() {
        if route.is_empty() {
            continue; // Empty routes are valid
        }
        
        // Constraint 3: Check capacity constraint
        check_capacity_constraint(route, nurse_idx, instance, &mut result);
        
        // Constraints 1, 2, 4: Check time constraints (depot start/end, time windows)
        check_time_constraints(route, nurse_idx, instance, &mut result);
    }
    
    result
}

/// Checks that each patient is visited exactly once
fn check_patient_coverage(solution: &Vec<Vec<usize>>, instance: &Instance, result: &mut ValidationResult) {
    let mut visited_patients = HashSet::new();
    let total_patients = instance.patients.len();
    
    // Collect all visited patients
    for (nurse_idx, route) in solution.iter().enumerate() {
        for &patient_id in route {
            if visited_patients.contains(&patient_id) {
                result.add_violation(format!(
                    "Patient {} is visited more than once", patient_id
                ));
            }
            visited_patients.insert(patient_id);
            
            // Check if patient ID is valid
            if !instance.patients.contains_key(&patient_id.to_string()) {
                result.add_violation(format!(
                    "Route {}: Invalid patient ID {}", nurse_idx, patient_id
                ));
            }
        }
    }
    
    // Check if all patients are visited
    for patient_id in 1..=total_patients {
        if !visited_patients.contains(&patient_id) {
            result.add_violation(format!(
                "Patient {} is not visited by any nurse", patient_id
            ));
        }
    }
}

/// Checks capacity constraint for a route
fn check_capacity_constraint(route: &Vec<usize>, nurse_idx: usize, instance: &Instance, result: &mut ValidationResult) {
    let mut total_demand = 0.0;
    
    for &patient_id in route {
        if let Some(patient) = instance.patients.get(&patient_id.to_string()) {
            total_demand += patient.demand;
        }
    }
    
    let capacity = instance.capacity_nurse as f64;
    
    if total_demand > capacity {
        result.add_violation(format!(
            "Route {}: Total demand ({:.1}) exceeds nurse capacity ({:.1})", 
            nurse_idx, total_demand, capacity
        ));
    }
}

/// Checks time constraints for a route (depot start/end times and patient time windows)
fn check_time_constraints(route: &Vec<usize>, nurse_idx: usize, instance: &Instance, result: &mut ValidationResult) {
    let mut current_time = 0.0; // Constraint 1: Route starts at depot at time 0
    let mut current_location = 0; // Start at depot (index 0)
    
    for &patient_id in route {
        if let Some(patient) = instance.patients.get(&patient_id.to_string()) {
            // Travel to patient
            let travel_time = instance.travel_times[current_location][patient_id];
            current_time += travel_time;
            
            // Wait if arriving before time window starts
            if current_time < patient.start_time {
                current_time = patient.start_time;
            }
            
            // Provide care
            current_time += patient.care_time;
            
            // Constraint 4: Check if service is within time window
            if current_time > patient.end_time {
                result.add_violation(format!(
                    "Route {}: Patient {} visited at time {:.2}, which is after end time {:.2}",
                    nurse_idx, patient_id, current_time, patient.end_time
                ));
            }
            
            current_location = patient_id;
        }
    }
    
    // Travel back to depot
    let return_travel_time = instance.travel_times[current_location][0];
    current_time += return_travel_time;
    
    // Constraint 2: Check if return time is within depot constraint
    if current_time > instance.depot.return_time {
        result.add_violation(format!(
            "Route {}: Returns to depot at time {:.2}, which is after depot return time {:.2}",
            nurse_idx, current_time, instance.depot.return_time
        ));
    }
}

/// Prints the validation result in a readable format
pub fn print_validation_result(result: &ValidationResult) {
    if result.is_feasible {
        println!("✓ Solution is FEASIBLE");
        println!("  All constraints are satisfied.");
    } else {
        println!("✗ Solution is INFEASIBLE");
        println!("  Found {} constraint violation(s):", result.violations.len());
        for (i, violation) in result.violations.iter().enumerate() {
            println!("  {}. {}", i + 1, violation);
        }
    }
}

/// Validates a solution and prints the result
pub fn validate_and_print(solution: &Vec<Vec<usize>>, instance: &Instance) -> bool {
    let result = validate_solution(solution, instance);
    print_validation_result(&result);
    result.is_feasible
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::parse_data::parse_data;
    
    #[test]
    fn test_empty_solution() {
        let instance = parse_data("src/data/train/train_0.json");
        let solution = vec![Vec::new(); instance.nurses.len()];
        let result = validate_solution(&solution, &instance);
        
        // Should be infeasible because patients are not visited
        assert!(!result.is_feasible);
        assert!(result.violations.len() > 0);
    }
    
    #[test]
    fn test_duplicate_patient() {
        let instance = parse_data("src/data/train/train_0.json");
        let mut solution = vec![Vec::new(); instance.nurses.len()];
        solution[0] = vec![1, 2, 3];
        solution[1] = vec![3, 4, 5]; // Patient 3 appears twice
        
        let result = validate_solution(&solution, &instance);
        
        assert!(!result.is_feasible);
        assert!(result.violations.iter().any(|v| v.contains("visited more than once")));
    }
}
