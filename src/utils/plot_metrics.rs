use plotters::prelude::*;

/// Plots the average, minimum, and maximum fitness over generations across all islands
///
/// # Arguments
///
/// * `island_results` - A vector of pairs `(island_id, fitness_history)`, where
///   `fitness_history` is a vector containing the best fitness value at each generation for that island
pub fn plot_fitness(island_results: &Vec<(usize, Vec<f64>)>) {
    // Determine the number of generations (assumes all islands have the same history length)
    let generations = if let Some((_, history)) = island_results.first() {
        history.len()
    } else {
        println!("No island results to plot.");
        return;
    };

    // Prepare vectors for average, min, and max fitness per generation.
    let mut avg_history: Vec<(f64, f64)> = Vec::with_capacity(generations);
    let mut min_history: Vec<(f64, f64)> = Vec::with_capacity(generations);
    let mut max_history: Vec<(f64, f64)> = Vec::with_capacity(generations);

    for gen in 0..generations {
        let mut sum = 0.0;
        let mut min_fit = f64::INFINITY;
        let mut max_fit = f64::NEG_INFINITY;
        for &(_, ref history) in island_results {
            let fit = history[gen];
            sum += fit;
            if fit < min_fit { min_fit = fit; }
            if fit > max_fit { max_fit = fit; }
        }
        let avg = sum / (island_results.len() as f64);
        avg_history.push((gen as f64, avg));
        min_history.push((gen as f64, min_fit));
        max_history.push((gen as f64, max_fit));
    }

    // Determines the global min and max for the y-axis plot
    let global_min = min_history.iter().map(|&(_, val)| val).fold(f64::INFINITY, f64::min);
    let global_max = max_history.iter().map(|&(_, val)| val).fold(f64::NEG_INFINITY, f64::max);
    // Ensures the minimum is positive for logarithmic scaling (adjust if necessary)
    let log_min = if global_min <= 0.0 { 1e-6 } else { global_min };

    let output_path = "output/fitness_plot.png";
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Build the chart with a logarithmic y-axis
    let mut chart = ChartBuilder::on(&root)
        .caption("Islands' Fitness Summary over Generations", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..(generations as f64), (log_min..global_max).log_scale())
        .unwrap();

    chart.configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness (log scale)")
        .draw()
        .unwrap();

    // Draws the average, minimum, and maximum curves.
    chart.draw_series(LineSeries::new(avg_history, &BLUE)).unwrap()
         .label("Avg")
         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    chart.draw_series(LineSeries::new(min_history, &RED)).unwrap()
         .label("Min")
         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart.draw_series(LineSeries::new(max_history, &GREEN)).unwrap()
         .label("Max")
         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Fitness plot saved as {}", output_path);
}
