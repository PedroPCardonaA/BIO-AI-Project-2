use crate::structs::{instance::Instance, nurse::Nurse};

/// Calculates the total cost of a nurse's route by summarizing travel time and penalty costs.
/// 
/// The cost comprises of:
/// - The total travel time from the depot through all patients and back to the depot.
/// - Penalties incurred for arriving after a patient's allowed end time, for exceeding the nurse's capacity,
///   and for returning to the depot later than allowed.
/// 
/// Penalty factors are defined as follows:
/// - A late penalty is applied per time unit beyond a patient's end time.
/// - A capacity penalty is applied per unit of load exceeding the nurse's capacity.
/// - A return penalty is applied per time unit if the nurse returns past the depot's return time.
/// 
/// # Parameters
/// - `route`: A vector of patient IDs representing the order in which patients are visited (the depot is implicit).
/// - `nurse`: The nurse assigned to the route, providing capacity constraints.
/// - `instance`: The problem instance containing travel times, patient data, and depot details.
/// 
/// # Returns
/// A `f64` value representing the sum of the travel time and all applicable penalties.
pub fn calculated_cost(route: &Vec<usize>, nurse: &Nurse, instance: &Instance) -> f64 {

    // STEP 1: Initialize variables.
    // Penalty factors                        // Description
    let penalty_factor_late = 55.0;      // Penalty per time unit past patient's end time.
    let penalty_factor_capacity = 4.0;   // Penalty per unit of load exceeding capacity.
    let penalty_factor_return = 3.0;     // Penalty per time unit past depot return time.

    // State variables                     
    let mut current_time = 0.0;          // Tracks cumulative time (travel, waiting, and service).
    let mut travel_time_sum = 0.0;       // Sum of travel times for the route.
    let mut total_penalty = 0.0;         // Accumulated penalties.
    let mut current_load = 0;            // Tracks current nurse load.
    let mut last_location = 0;         // Start at the depot (assumed index 0).

    // STEP 2: Process each patient in the route sequentially.
    for patient_id in route {

        // STEP 2.1: Add travel time from the last location to the current patient.
        let travel_time = instance.travel_times[last_location][*patient_id];
        travel_time_sum += travel_time;
        current_time += travel_time;

        // STEP 2.2: Retrieve current patient details.
        let patient = &instance.patients[&patient_id.to_string()];

        // STEP 2.3: If arriving before the patient's start time, wait until the service can begin.
        if current_time < patient.start_time {
            current_time = patient.start_time;
        }

        // STEP 2.4: Add the care (service) time required for the patient.
        current_time += patient.care_time;

        // STEP 2.5: If service is provided after the patient's end time, apply a late penalty.
        if current_time > patient.end_time {
            total_penalty += penalty_factor_late * (current_time - patient.end_time);
        }

        // STEP 2.6: Update the nurse's load with the patient's demand and apply a capacity penalty if exceeded.
        current_load += patient.demand as u32;
        if current_load > nurse.get_capacity() {
            total_penalty += penalty_factor_capacity * ((current_load - nurse.get_capacity()) as f64);
        }

        // STEP 2.7: Update the last visited location to the current patient.
        last_location = *patient_id;
    }

    // STEP 3: Add travel time from the last patient back to the depot.
    let travel_time = instance.travel_times[last_location][0];
    travel_time_sum += travel_time;
    current_time += travel_time;

    // STEP 4: If the nurse returns later than allowed, apply a return penalty.
    if current_time > instance.depot.return_time {
        total_penalty += penalty_factor_return * (current_time - instance.depot.return_time);
    }

    // STEP 5: Return the sum of travel time and total penalties as the route's cost.
    travel_time_sum + total_penalty
}


/// Computes the total fitness of a solution by summing the cost of each nurse's route.
/// 
/// For a given solution—represented as a vector of routes (one per nurse)—this function calculates
/// the cost for each route using `calculated_cost` and returns the cumulative cost as the solution's fitness.
/// This means that it is a cost function that should be minimized.
/// 
/// # Parameters
/// - `solution`: A vector of routes, where each route is a vector of patient IDs.
/// - `instance`: The problem instance containing necessary data such as nurses, travel times, and patient details.
/// 
/// # Returns
/// A `f64` value representing the total cost (fitness) of the solution.
pub fn fitness(solution: &Vec<Vec<usize>>, instance: &Instance) -> f64 {
    let mut total_cost = 0.0;
    // For each nurse and its corresponding route, sum the cost.
    for (nurse, route) in instance.nurses.iter().zip(solution.iter()) {
        total_cost += calculated_cost(route, nurse, instance);
    }
    total_cost
}
