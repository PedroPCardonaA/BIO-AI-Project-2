use plotters::{coord::types::RangedCoordf64, prelude::*};
use std::{collections::HashMap, f64::consts::PI};

use crate::structs::{depot::Depot, patient::Patient};

/// Plots the nurse routing solution on a map and saves it as a PNG image.
/// 
/// This function generates a visual representation of the solution by plotting the depot, patient locations,
/// and nurse routes. The patient coordinates are scaled relative to the depot using a scale factor.
/// Unique colors are generated for each nurse route via an HSL-to-RGB conversion, and arrows are drawn along
/// the routes to indicate direction. A legend is added to identify each nurse's route.
/// 
/// # Parameters
/// - `solution`: A reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
/// - `patients`: A reference to a HashMap mapping patient IDs (as strings) to their corresponding `Patient` structs.
/// - `depot`: A reference to the `Depot` struct containing the depot's coordinates and return time.
/// 
/// # Remarks
/// The generated map is saved as "output/solution.png".
pub fn plot_map(solution: &Vec<Vec<usize>>, patients: &HashMap<String, Patient>, depot: &Depot) {
    let output_path = "output/solution.png";
    let root = BitMapBackend::new(output_path, (900, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Scale factor for the distances.
    let scale_factor = 1.5;

    // Helper function: scales a point relative to the depot.
    fn scale_point(x: f64, y: f64, depot: &Depot, factor: f64) -> (f64, f64) {
        (depot.x_coord + (x - depot.x_coord) * factor,
         depot.y_coord + (y - depot.y_coord) * factor)
    }

    // Compute scaled bounds for the chart.
    let (scaled_depot_x, scaled_depot_y) = (depot.x_coord, depot.y_coord);
    let mut min_x = scaled_depot_x;
    let mut max_x = scaled_depot_x;
    let mut min_y = scaled_depot_y;
    let mut max_y = scaled_depot_y;
    for patient in patients.values() {
        let (scaled_x, scaled_y) = scale_point(patient.x_coord, patient.y_coord, depot, scale_factor);
        if scaled_x < min_x { min_x = scaled_x; }
        if scaled_x > max_x { max_x = scaled_x; }
        if scaled_y < min_y { min_y = scaled_y; }
        if scaled_y > max_y { max_y = scaled_y; }
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("Nurse Routing Solution", ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Generate unique colors for each nurse using HSL color space.
    let num_nurses = solution.len();
    let colors: Vec<RGBColor> = (0..num_nurses)
        .map(|i| {
            let hue = 360.0 * (i as f64 / num_nurses as f64);
            let (r, g, b) = hsl_to_rgb(hue, 0.8, 0.5);
            RGBColor(r, g, b)
        })
        .collect();

    // Draw the depot.
    let depot_point = scale_point(depot.x_coord, depot.y_coord, depot, scale_factor);
    chart
        .draw_series(std::iter::once(Circle::new(depot_point, 5, BLACK.filled())))
        .unwrap();

    // For each nurse, draw its route.
    for (nurse_id, route) in solution.iter().enumerate() {
        let color = colors[nurse_id];

        // Build the path: start at the depot, then visit each patient (scaled), and return to the depot.
        let mut path_points = vec![depot_point];
        for patient_id in route {
            if let Some(patient) = patients.get(&patient_id.to_string()) {
                let scaled_coords = scale_point(patient.x_coord, patient.y_coord, depot, scale_factor);
                path_points.push(scaled_coords);
            }
        }
        path_points.push(depot_point);

        // Draw the route as a line.
        chart
            .draw_series(LineSeries::new(path_points.iter().copied(), &color))
            .unwrap();

        // Draw arrows along the route.
        for window in path_points.windows(2) {
            if let [start, end] = *window {
                let angle = ((end.1 - start.1).atan2(end.0 - start.0)).to_degrees();
                draw_arrow(&mut chart, end, angle, &color);
            }
        }

        // Draw patient markers with a smaller radius.
        // Skip the first and last points (which are the depot) when drawing patient markers.
        for &point in path_points.iter().skip(1).take(path_points.len() - 2) {
            chart
                .draw_series(std::iter::once(Circle::new(point, 3, color.filled())))
                .unwrap();
        }
    }

    // Draw a legend in the top-right corner.
    let legend_x = max_x - (max_x - min_x) * 0.2;
    let legend_y = max_y - (max_y - min_y) * 0.05;
    for (i, color) in colors.iter().enumerate() {
        let legend_text = format!("Nurse {}", i + 1);
        chart.draw_series(std::iter::once(Text::new(
            legend_text,
            (legend_x, legend_y - i as f64 * 10.0),
            TextStyle::from(("sans-serif", 15).into_font()).color(color),
        )))
        .unwrap();
    }

    root.present().unwrap();
    println!("Solution diagram saved as {}", output_path);
}

/// Converts a color from HSL (Hue, Saturation, Lightness) to RGB (Red, Green, Blue).
/// 
/// # Parameters
/// - `h`: The hue angle in degrees.
/// - `s`: The saturation component (0.0 to 1.0).
/// - `l`: The lightness component (0.0 to 1.0).
/// 
/// # Returns
/// A tuple `(r, g, b)` where each component is an 8-bit unsigned integer representing the corresponding RGB value.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Draws a small arrow at the end of a route segment to indicate direction.
/// 
/// This function draws two short line segments forming an arrowhead at the specified end point of a route segment.
/// The arrow is oriented based on the provided angle, and its size is defined internally.
/// 
/// # Parameters
/// - `chart`: A mutable reference to the chart context used for drawing.
/// - `end`: A tuple `(f64, f64)` representing the end point of the route segment.
/// - `angle`: The angle (in degrees) of the route segment, used to orient the arrow.
/// - `color`: A reference to an `RGBColor` specifying the color of the arrow.
fn draw_arrow(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    end: (f64, f64),
    angle: f64,
    color: &RGBColor,
) {
    // Reduced arrow size.
    let arrow_length = 2.0;
    let angle_rad = angle.to_radians();

    let arrow_x1 = end.0 - arrow_length * (angle_rad + PI / 6.0).cos();
    let arrow_y1 = end.1 - arrow_length * (angle_rad + PI / 6.0).sin();
    let arrow_x2 = end.0 - arrow_length * (angle_rad - PI / 6.0).cos();
    let arrow_y2 = end.1 - arrow_length * (angle_rad - PI / 6.0).sin();

    chart
        .draw_series(LineSeries::new(vec![end, (arrow_x1, arrow_y1)], color))
        .unwrap();
    chart
        .draw_series(LineSeries::new(vec![end, (arrow_x2, arrow_y2)], color))
        .unwrap();
}