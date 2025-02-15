use crate::structs::instance::Instance;

pub fn fitness(solution: &Vec<Vec<usize>>, instance: &Instance) -> f64 {
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