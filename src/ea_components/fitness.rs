use crate::structs::instance::Instance;

pub fn fitness_1(solution: &Vec<Vec<usize>>, instance: &Instance) -> f64 {
    let mut total_travel_time = 0.0;
    let mut total_penalty = 0.0;
    let penalty_factor = 1.0; // Higher value means higher penalty for returning too late
    let penalty_factor_time = 5.0; // Higher value means higher penalty for traveling too much between patients
    let penalty_factor_violation = 15.0; // Higher value means higher penalty for violating the constraints

    // Calculate the total travel time for each nurse
    let mut nurses = instance.nurses.clone();
    for (nurse, route) in nurses.iter_mut().zip(solution.iter()) {
        let mut last_patient = 0; // The depot is the first patient

        //println!("New nurse");

        // Calculate the travel time and capacity for each patient in the route
        for patient_id in route {
            let mut wait_time = 0.0;
            let patient = &instance.patients[&patient_id.to_string()];
            // Print the nurse and patient ID
            //println!("Last patient: {:?}, Current patient: {:?}", last_patient, patient_id);

            // Calculate the travel time from the last patient to the current patient
            let travel_time = instance.travel_times[last_patient][*patient_id];

            // Add the travel time as a penalty, since then nurses will be penalized for traveling too much between patients
            total_penalty += travel_time * penalty_factor_time;

            // Check if the nurse visits the patient too early
            if patient.start_time > (nurse.get_current_travel_time() + travel_time) {
                wait_time = patient.start_time - (nurse.get_current_travel_time() + travel_time);
            }

            // Add the travel time to the nurse's current travel time
            nurse.set_current_travel_time(nurse.get_current_travel_time() + travel_time + patient.care_time + wait_time);

            // Check if the nurse visits the patient too late
            if patient.end_time < nurse.get_current_travel_time() {
                total_penalty += penalty_factor_violation * (nurse.get_current_travel_time() - patient.end_time);
            }

            // Add the patient's demand to the nurse's current load
            nurse.set_current_load(nurse.get_current_load() + patient.demand as u32);

            // Set the current patient as the last patient
            last_patient = *patient_id;
        }

        // Add the travel time from the last patient to the depot
        let travel_time = instance.travel_times[last_patient][0];
        nurse.set_current_travel_time(nurse.get_current_travel_time() + travel_time);

        // Check if the nurses capacity is exceeded
        if nurse.get_current_load() as f64 > nurse.get_capacity() as f64 {
            total_penalty += penalty_factor_violation * (nurse.get_current_load() as f64 - nurse.get_capacity() as f64);
        }

        // Check if the nurse returns to the depot too late
        if nurse.get_current_travel_time() > instance.depot.return_time {
            total_penalty += penalty_factor * (nurse.get_current_travel_time() - instance.depot.return_time);
        }

        // Add the nurse's travel time to the total travel time
        total_travel_time += nurse.get_current_travel_time();
    }

    total_travel_time + total_penalty
}
pub fn fitness(solution: &Vec<Vec<usize>>, instance: &Instance) -> f64 {
    let mut total_travel_time = 0.0;
    let mut total_penalty = 0.0;

    // Unique penalty factors for each constraint violation.
    let penalty_factor_late = 1.7;      // Penalty per time unit past patient's end time.
    let penalty_factor_capacity = 1.0;  // Penalty per unit of load exceeding capacity.
    let penalty_factor_return = 1.0;    // Penalty per time unit past depot return time.

    // For each nurse and its corresponding route in the solution.
    for (nurse, route) in instance.nurses.iter().zip(solution.iter()) {
        let mut current_time = 0.0;      // Includes travel, waiting, and care time (for constraint checking).
        let mut travel_time_sum = 0.0;   // Sum of travel times only (objective).
        let mut current_load = 0;
        let mut last_location = 0;       // Start at the depot (assumed index 0).

        for patient_id in route {
            // Travel from last location to current patient.
            let travel_time = instance.travel_times[last_location][*patient_id];
            travel_time_sum += travel_time;
            current_time += travel_time;

            // Retrieve patient details.
            let patient = &instance.patients[&patient_id.to_string()];

            // If arriving earlier than the start time, wait until the start.
            if current_time < patient.start_time {
                current_time = patient.start_time;
            }

            // Add the care (service) time.
            current_time += patient.care_time;

            // If arriving later than the patient's end time, add a penalty.
            if current_time > patient.end_time {
                total_penalty += penalty_factor_late * (current_time - patient.end_time);
            }



            // Update nurse's load and check capacity.
            current_load += patient.demand as u32;
            if current_load > nurse.get_capacity() {
                total_penalty += penalty_factor_capacity * ((current_load - nurse.get_capacity()) as f64);
            }

            // Update last visited location.
            last_location = *patient_id;
        }

        // Travel from last patient back to the depot.
        let travel_time = instance.travel_times[last_location][0];
        travel_time_sum += travel_time;
        current_time += travel_time;

        // If the nurse returns too late, add a penalty.
        if current_time > instance.depot.return_time {
            total_penalty += penalty_factor_return * (current_time - instance.depot.return_time);
        }

        // Accumulate the travel time for the overall solution.
        total_travel_time += travel_time_sum;
    }

    total_travel_time + total_penalty
}