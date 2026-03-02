## COMPUTER GRAPHICS

C VERSION

<!-- image -->

DONALD HEARN • M. PAULINE BAKER

Contents

PREFACE

1

1-1

1-2

1-3

1-4

1-5

1-6

1-7

1-8

2

2-1

## Contents

Computer-Aided Design

Presentation Graphics xvii

2

4

11

2-2

2-3

Stereoscopic and Virtual-Reality

Systems

Raster-Scan Systems

Video Controller

Raster-Scan Display Processor

Random-Scan Systems

50

53

55

56

| PREFACE   | PREFACE                           | xvii   |     | Stereoscopicand Virtual-Reality Systems                             |
|-----------|-----------------------------------|--------|-----|---------------------------------------------------------------------|
| 1         | A Survey of Computer Graphics     | 2      | 2-2 | Raster-Scan System!; Video Controller Raster-Scan Display Processor |
|           | Computer-Aided Design             |        | 2-3 | Random-Scan Systems                                                 |
|           | Presentation Graphics             | 'I     | 2-4 | Graphics Monitors and Workstations                                  |
|           | Computer Art                      | l 3    | 2-5 | Input Devices                                                       |
|           | Entertainment                     | 18     |     | Keyboards                                                           |
|           | Education and Training            | 21     |     | Mouse                                                               |
|           | Visualization                     | 25     |     | Trackballand Spaceball                                              |
|           | Image Processing                  | 32     |     | Joysticks                                                           |
|           | Graphical User Interfaces         | 34     |     | Data Glove                                                          |
|           |                                   |        |     | Digitizers                                                          |
|           |                                   |        |     | Image Scanners                                                      |
|           |                                   |        |     | Touch Panels                                                        |
| 2         | Overview of Graphics systems      | 35     | 2-6 | Voice Systems Hard-Copy Devices                                     |
| 2-1       | VideoDisplayDevices               | 36     | 2-7 | Graphics Software                                                   |
|           | Refresh Cathode-Ray Tubes         | 37     |     | Coordinate Representations                                          |
|           | Raster-Scan Displays              | 40     |     | Graphics Functions                                                  |
|           | Random-Scan Displays              | 41     |     | SoftwareStandards                                                   |
|           | Color CRTMonitors                 | 42     |     | PHIGSWorkstations                                                   |
|           | Direct-View Storage Tubes         | 4.5    |     | Summary                                                             |
|           | Flat-Panel Displays               | 45     |     | References                                                          |
|           | Three-Dimensional Viewing Devices | 49     |     | Exercises                                                           |

| 3   |                                                       |    | Summary                              |     |
|-----|-------------------------------------------------------|----|--------------------------------------|-----|
|     | Outout Primitives                                     | 83 | Applications References              |     |
|     | Points and Lines                                      |    | Exercises                            |     |
|     | Line-DrawingAlgorithms DDAAlgorithm                   |    |                                      |     |
|     | Bresenham's Line Algorithm Parallel Line Algorithms   |    |                                      |     |
|     |                                                       |    | Attributes of Output                 |     |
|     | Loading the Frame Buffer                              |    | Primitives                           | 143 |
|     | Line Function                                         |    |                                      |     |
|     | Circle-GeneratingAlgorithms                           |    | Line Attributes                      |     |
|     | Properties of Circles                                 |    | LineType                             |     |
|     | Midpoint Circle Algorithm                             |    | LineWidth                            |     |
|     | Ellipse-GeneratingAlgorithms                          |    | Pen and Brush Options                |     |
|     | Properties of Ellipses                                |    | LineColor                            |     |
|     | Midpoint Ellipse Algorithm                            |    | Curve Attributes                     |     |
|     | Other Curves                                          |    | Color and Grayscale Levels           |     |
|     | Conic Sections                                        |    | Color Tables                         |     |
|     | Polynomials and SplineCurves                          |    | Grayscale                            |     |
|     | Parallel Curve Algorithms                             |    | Area-Fill Attributes                 |     |
|     | Curve Functions                                       |    | Fill Styles                          |     |
|     | Pixel Addressing                                      |    | Pattern Fill                         |     |
|     | and Object Geometry                                   |    | SoftFill                             |     |
|     | Screen Grid Coordinates                               |    | Character Attributes                 |     |
|     | Maintaining Geometric Properties of Displayed Objects |    | Text Attributes Marker Attributes    |     |
|     | Filled-Area Primitives                                |    | Bundled Attributes                   |     |
|     | Scan-LinePolygon F i l l Algorithm                    |    | Bundled Line Attributes              |     |
|     | Inside-Outside Tests                                  |    | Bundled Area-Fi Attributes           |     |
|     | Scan-Line Fill of Curved Boundary Areas               |    | Bundled Text Attributes              |     |
|     | Boundary-Fill Algorithm                               |    | Bundled Marker Attributes            |     |
|     |                                                       |    | Inquiry Functions                    |     |
|     | Flood -Fill Algorithm                                 |    | Antialiasing                         |     |
|     | Fill-Area Functions Cell Array                        |    | Supersampling Straight Line Segments |     |
|     | Character Generation                                  |    | Pixel-Weighting Masks                |     |

5

5-1

5-2

5-3

5-4

5-5

Area Sampling Straight Line

Segments

Filtering Techniques

Pixel Phasing

Summary

Exercises

Rotations

Shear

Systems

174

174

175

Contents

Affine Transformations

Transformation Functions

Summary

<!-- image -->

208

208

210

212

213

|     | Area Sampling Straight Line                                   |      | 5-6   | Aff ine Transformations                             | 208   |
|-----|---------------------------------------------------------------|------|-------|-----------------------------------------------------|-------|
|     | Segments                                                      | 174  | 5-7   | Transformation Functions                            | 208   |
|     | Filtering Techniques                                          | 174  | 5-8   | Raster Methods for Transformations                  | 210   |
|     | Pixel Phasing                                                 | 1 75 |       | Summary                                             | 212   |
|     | Compensating for Line lntensity                               |      |       | References                                          | 213   |
|     | Differences                                                   | 1 75 |       |                                                     |       |
|     | Antialiasing Area Boundaries                                  | 176  |       | Exercises                                           | 213   |
|     | Summary                                                       |      |       |                                                     |       |
|     | References                                                    |      | 6     | Two-Dimensional                                     |       |
|     | Exercises                                                     | 180  |       | Viewing                                             | 21 6  |
|     |                                                               |      | 6-1   | The Viewing Pipeline                                |       |
|     | Two-DimensionalGeometric                                      |      | 6-2   | Viewing Coordinate Reference Frame                  |       |
| 5   | Transformations                                               | 183  | 6-3   | Window-teviewport Coordinate Transformation         |       |
| 5-1 | Basic Transformations Translation                             |      |       | Two-DimensionalWewing Functions Clipping Operations |       |
|     | Rotation                                                      |      |       | Point Clipping                                      |       |
|     | Scaling                                                       |      |       | Line Clipping                                       |       |
| 5-2 | Matrix Representations                                        |      |       | Cohen-Sutherland Line Clipping                      |       |
|     | and Homogeneous Coordinates                                   |      |       | Liang-Barsky Line Clipping                          |       |
| 5-3 | Composite Transformations                                     |      |       | Nicholl-Lee-Nicholl Line Clipping                   |       |
|     | Translations                                                  |      |       | Line Clipping Using Nonrectangular Clip Windows     |       |
|     | Rotations                                                     |      |       | Splitting Concave Polygons                          |       |
|     | Scalings                                                      |      |       |                                                     |       |
|     | General Pivot-Point Rotation                                  |      |       | Polygon Clipping Sutherland-Hodgernan               |       |
|     | General Fixed-Point Scaling                                   |      |       | Polygon Clipping                                    |       |
|     | General Scaling Directions                                    |      |       | Weiler-Atherton Polygon Clipping                    |       |
|     | Concatenation Properties                                      |      |       | Other Polygon-Clipping Algorithms                   |       |
|     | General CompositeTransformations and Computational Efficiency |      |       | Curve Clipping                                      |       |
| 5-4 | Other Transformations                                         |      |       | Text Clipping                                       |       |
|     | Reflection Shear                                              |      |       | Exterior Clipping Summary                           |       |
| 5-5 | TransformationsBetween Coordinate                             | 205  |       | References                                          |       |
|     | Systems                                                       |      |       | Exercises                                           |       |

5-6

5-7

5-8

210

6

7

7-1

7-2

7-3

7-4

8

8-1

x

Contents

Structures and Hierarchical

Modeling

Structure Concepts

250

250

Accommodating Multiple

Skill Levels

Consistency

Backup and Error Handling

273

274

274

274

| 7   | Structures and Hierarchical                    | Structures and Hierarchical   | Structures and Hierarchical   | Accommodating Multiple                                  | Accommodating Multiple   |
|-----|------------------------------------------------|-------------------------------|-------------------------------|---------------------------------------------------------|--------------------------|
|     | Modeling                                       | 250                           |                               | Skill Levels Consistency                                |                          |
| 7-1 | Structure Concepts                             | 250                           |                               | Minimizing Memorization                                 |                          |
|     | Basic Structure Functions                      | 250                           |                               | Backup and Error Handling                               |                          |
|     | Setting Structure Attributes                   | 253                           |                               | Feedback                                                |                          |
| 7-2 | Editing Structures                             | 254                           | 8-2                           | lnput of Graphical Data                                 |                          |
|     | Structure Lists and the Element Pointer        | 255                           |                               | Logical Classification of Input Devices                 |                          |
|     | Setting the Edit Mode                          | 250                           |                               | Locator Devices                                         |                          |
|     | Inserting Structure Elements                   | 256                           |                               | Stroke Devices                                          |                          |
|     | Replacing Structure Elements                   | 257                           |                               | String Devices                                          |                          |
|     | Deleting Structure Elements                    | 257                           |                               | Valuator Devices                                        |                          |
|     | LabelingStructure Elements                     | 258                           |                               | Choice Devices                                          |                          |
|     | Copying Elements fromOneStructure              |                               |                               | Pick Devices                                            |                          |
|     | to Another                                     | 260                           | 8-3                           | lnput Functions                                         |                          |
| 7-3 | Basic Modeling Concepts                        | 260                           |                               | Input Modes                                             |                          |
|     | Mode1Representations                           | 261                           |                               | Request Mode                                            |                          |
|     | Symbol Hierarchies                             | 262                           |                               | Locator and Stroke Input in Request Mode                |                          |
|     | Modeling Packages. Hierarchical Modeling       | 263                           |                               | String Input in Request Mode                            |                          |
| 7-4 | with Structures                                | 265                           |                               | Valuator Input in Request Mode                          |                          |
|     | Local Coordinates and Modeling Transformations | 265                           |                               | Choice lnput in Request Mode Pick Input in Request Mode |                          |
|     | Modeling Transformations                       | 266                           |                               | SampleMode                                              |                          |
|     | Structure Hierarchies                          | 266                           |                               | Event Mode                                              |                          |
|     | Summary                                        | 268                           |                               | Concurrent Use of Input Modes                           |                          |
|     | References                                     | 269                           | 8-4                           | Initial Values for Input-Device Parameters              |                          |
|     | Exercises                                      |                               |                               |                                                         |                          |
|     |                                                | 2 69                          |                               | lnteractive Picture-Construction Techniques             |                          |
|     | Graphical User Interfaces                      |                               | 8-5                           | Basic Positioning Methods                               |                          |
|     | and Interactive lnput                          |                               |                               | Constraints Grids                                       |                          |
| 8   | Methods                                        | 271                           |                               | Gravity Field                                           |                          |
|     |                                                |                               |                               | Rubber-BandMethods                                      |                          |
| 8-1 | The User Dialogue                              |                               |                               | Dragging                                                |                          |
|     | Windows and Icons                              |                               |                               | Painting and Drawing                                    |                          |

Minimizing Memorization

8-6

9

9-1

9-2

Packages

10

10-1

10-2

10-3

Virtual-Reality Environments

Summary

References

Exercises

Views

292

293

294

294

10-4

10-5

10-6

Contents

Superquadrics

Superellipse

Superellipsoid

Blobby Objects

312

312

313

314

315

| 8-6                            | Virtual-Reality Environments       | 292                                | 10-4   | Superquadrics                                         |
|--------------------------------|------------------------------------|------------------------------------|--------|-------------------------------------------------------|
|                                | Summary                            | 233                                |        | Superellipse                                          |
|                                | References                         | 294                                |        | Superellipsoid                                        |
|                                | Exercises                          | 294                                | 10-5   | Blobby Objects                                        |
|                                |                                    |                                    | 10-6   | Spline Representations                                |
|                                |                                    |                                    |        | Interpolation and Approximation Splines               |
| 9                              | Three-Dimensional Concepts         | 296                                |        | Parametric Continuity Conditions Geometric Continuity |
| 9-1                            | Three-DimensionalDisplay Methods   | Three-DimensionalDisplay Methods   |        | Conditions                                            |
|                                | Parallel Projection                | Parallel Projection                |        | SplineSpecifications                                  |
|                                | Perspective Projection             | Perspective Projection             |        | Cubic Spline Interpolation Methods                    |
|                                | Visible Line and Surface           | Visible Line and Surface           |        |                                                       |
|                                |                                    |                                    |        | Natural Cubic Splines                                 |
|                                | Identification                     | Identification                     |        | Hermite Interpolation                                 |
|                                | Surface Rendering                  | Surface Rendering                  |        | Cardinal Splines                                      |
|                                | Exploded and Cutaway Views         | Exploded and Cutaway Views         |        | Kochanek-BartelsSplines                               |
|                                | Three-Dimensional and Stereoscopic | Three-Dimensional and Stereoscopic |        | Bezier Curves and Surfaces                            |
|                                | Views                              | Views                              |        | Bezier Curves                                         |
| 9-2                            | Three-DimensionalGraphics          | 302                                |        | Properties of Bezier Curves                           |
|                                |                                    |                                    |        | Design Techniques Using Bezier Curves                 |
|                                |                                    |                                    |        | Cubic Ezier Curves                                    |
|                                | Three-Dimensional                  | Three-Dimensional                  |        | Bezier Surfaces                                       |
|                                |                                    |                                    |        | B-SplineCurves and Surfaces B-Spline Curves           |
|                                |                                    |                                    |        | Uniform, Periodic B-Splines                           |
|                                |                                    |                                    |        | Cubic, Periodic €3-Splines                            |
| 10-1 Polygon Surfaces          | 10-1 Polygon Surfaces              | 10-1 Polygon Surfaces              |        | Open, Uniform B-Splines                               |
| Polygon Tables                 | Polygon Tables                     | Polygon Tables                     |        | Nonuniform 13-Splines                                 |
| Plane Equations                | Plane Equations                    | Plane Equations                    |        | B-SplineSurfaces                                      |
| Polygon Meshes                 | Polygon Meshes                     | Polygon Meshes                     |        | Beta-Splines                                          |
| 10-2 Curved Lines and Surfaces | 10-2 Curved Lines and Surfaces     | 10-2 Curved Lines and Surfaces     |        | Beta-SplineContinuity                                 |
| 10-3 Quadric Sutiaces          | 10-3 Quadric Sutiaces              | 10-3 Quadric Sutiaces              |        | Conditions                                            |
| Sphere                         | Sphere                             | Sphere                             |        | Cubic, Periodic Beta-Spline                           |
| Ellipsoid                      | Ellipsoid                          | Ellipsoid                          |        | Matrix Representation                                 |
| Torus                          | Torus                              | Torus                              |        | Rational Splines                                      |8-6

9

9-1

9-2

Packages

10

10-1

10-2

10-3

Virtual-Reality Environments

Summary

References

Exercises

Views

292

293

294

294

10-4

10-5

10-6

Contents

Superquadrics

Superellipse

Superellipsoid

Blobby Objects

312

312

313

314

315

| 8-6                            | Virtual-Reality Environments       | 292                                | 10-4   | Superquadrics                                         |
|--------------------------------|------------------------------------|------------------------------------|--------|-------------------------------------------------------|
|                                | Summary                            | 233                                |        | Superellipse                                          |
|                                | References                         | 294                                |        | Superellipsoid                                        |
|                                | Exercises                          | 294                                | 10-5   | Blobby Objects                                        |
|                                |                                    |                                    | 10-6   | Spline Representations                                |
|                                |                                    |                                    |        | Interpolation and Approximation Splines               |
| 9                              | Three-Dimensional Concepts         | 296                                |        | Parametric Continuity Conditions Geometric Continuity |
| 9-1                            | Three-DimensionalDisplay Methods   | Three-DimensionalDisplay Methods   |        | Conditions                                            |
|                                | Parallel Projection                | Parallel Projection                |        | SplineSpecifications                                  |
|                                | Perspective Projection             | Perspective Projection             |        | Cubic Spline Interpolation Methods                    |
|                                | Visible Line and Surface           | Visible Line and Surface           |        |                                                       |
|                                |                                    |                                    |        | Natural Cubic Splines                                 |
|                                | Identification                     | Identification                     |        | Hermite Interpolation                                 |
|                                | Surface Rendering                  | Surface Rendering                  |        | Cardinal Splines                                      |
|                                | Exploded and Cutaway Views         | Exploded and Cutaway Views         |        | Kochanek-BartelsSplines                               |
|                                | Three-Dimensional and Stereoscopic | Three-Dimensional and Stereoscopic |        | Bezier Curves and Surfaces                            |
|                                | Views                              | Views                              |        | Bezier Curves                                         |
| 9-2                            | Three-DimensionalGraphics          | 302                                |        | Properties of Bezier Curves                           |
|                                |                                    |                                    |        | Design Techniques Using Bezier Curves                 |
|                                |                                    |                                    |        | Cubic Ezier Curves                                    |
|                                | Three-Dimensional                  | Three-Dimensional                  |        | Bezier Surfaces                                       |
|                                |                                    |                                    |        | B-SplineCurves and Surfaces B-Spline Curves           |
|                                |                                    |                                    |        | Uniform, Periodic B-Splines                           |
|                                |                                    |                                    |        | Cubic, Periodic €3-Splines                            |
| 10-1 Polygon Surfaces          | 10-1 Polygon Surfaces              | 10-1 Polygon Surfaces              |        | Open, Uniform B-Splines                               |
| Polygon Tables                 | Polygon Tables                     | Polygon Tables                     |        | Nonuniform 13-Splines                                 |
| Plane Equations                | Plane Equations                    | Plane Equations                    |        | B-SplineSurfaces                                      |
| Polygon Meshes                 | Polygon Meshes                     | Polygon Meshes                     |        | Beta-Splines                                          |
| 10-2 Curved Lines and Surfaces | 10-2 Curved Lines and Surfaces     | 10-2 Curved Lines and Surfaces     |        | Beta-SplineContinuity                                 |
| 10-3 Quadric Sutiaces          | 10-3 Quadric Sutiaces              | 10-3 Quadric Sutiaces              |        | Conditions                                            |
| Sphere                         | Sphere                             | Sphere                             |        | Cubic, Periodic Beta-Spline                           |
| Ellipsoid                      | Ellipsoid                          | Ellipsoid                          |        | Matrix Representation                                 |
| Torus                          | Torus                              | Torus                              |        | Rational Splines                                      |

10-12

10-13

10-14

10-15

10-16

10-17

10-18

10-19

10-20

10-21

10-22

xii

402

404

| Contents                                                     |     |      | Visual Representations for Multivariate Data Fields   |                |
|--------------------------------------------------------------|-----|------|-------------------------------------------------------|----------------|
| Displaying Spline Curves                                     |     |      | Summary                                               | 404            |
| and Surfaces                                                 |     |      | References                                            | 404            |
| Homer's Rule                                                 |     |      | Exercises                                             | 404            |
| Forward-Difference Calculations                              |     |      |                                                       |                |
| Subdivision Methods                                          |     |      |                                                       |                |
| Sweep Representations                                        |     |      | Three-Dimensional                                     |                |
| Constructive Solid-Geometry Methods                          |     | 11   | Geometric and Modeling Transformations                | 407            |
| Octrees                                                      |     |      |                                                       |                |
| BSP Trees                                                    |     |      | Translation                                           | 408            |
| Fractal-Geometry Methods                                     |     |      | Rotation                                              | 409            |
| Fractal-Generation Procedures                                |     |      | Coordinate-Axes Rotations                             | 409            |
| Classification of Fractals Fractal Dimension                 |     |      | General Three-Dimensional Rotations                   | 41 3           |
| Geometric Construction of Deterministic Self-Similar         |     |      | Rotations with Quaternions Scaling                    | 419 420        |
| Fractals GeometricConstruction of Statistically Self-Similar |     |      |                                                       |                |
|                                                              |     |      | Other Transformat~ons                                 | 422            |
|                                                              |     |      | Reflections                                           | 422            |
| Fractals                                                     |     |      | Shears                                                | 423            |
| Affine Fractal-Construction Methods                          |     |      | Conlposite Transformations                            | 423            |
| Random Midpoint-Displacement                                 |     |      | Three-Dimens~onal Functions                           | Transformation |
| Methods                                                      |     |      | Modeling and Coordinate                               | 425            |
| Controlling Terrain Topography                               |     |      | Transformations                                       | 426            |
| Self-squaring Fractals                                       |     |      | Summary                                               | 429            |
| Self-inverse Fractals                                        |     |      | References                                            | 429            |
| Shape Grammarsand Other Procedural Methods                   |     |      | Exercises                                             | 430            |
| Particle Systems                                             |     |      |                                                       |                |
| Physically Based Modeling                                    |     |      | Three-Dimensional                                     |                |
| Visualization of Data Sets                                   |     | 12   | Viewing                                               | 43 1           |
| Visual Representations for Scalar Fields                     |     | 12-1 | Viewing Pipeline                                      | 432            |
| VisuaI Representations for Vector Fields                     |     | 12-2 | Viewing Coordinates Specifying the Virbw Plane        | 433 433        |
| Visual Representations for Tensor Fields -                   | 401 |      | Transformation from World to Viewing Coordinates      | 437            |

Contents

Conversion Between Spline

Representations

349

Visual Representations for Multivariate Data Fields

Summary

1

12-3

12-4

12-5

12-6

12-7

13

13-1

13-2

13-3

13-4

13-5

13-6

13-7

13-8

13-9

13-10

13-11

Projections

Parallel Projections

Perspective Projections

View Volumes and General

438

439

443

13-12

13-13

Contents

Wireframe Methods

Visibility-Detection Functions

Summary

## Contents

490

490

491

| Projections                                    |     | 1 3-12   | Wireframe Methods                         | 490   |
|------------------------------------------------|-----|----------|-------------------------------------------|-------|
| Parallel Projections                           |     | 13-1 3   | Visibility-DetectionFunctions             | 490   |
| Perspective IJrojections                       |     |          | Summary                                   | 49 1  |
| View Volumes and General                       |     |          | Keferences                                | 492   |
| Projection Transformations                     |     |          | Exercises                                 | 49 2  |
| General Parallel-Projection Transformations    |     |          |                                           |       |
| General Perspective-Projection Transformations |     |          | lllumination Models                       |       |
| Clipping                                       |     | 14       | and Surface-Rendering                     |       |
| Volumes                                        |     |          | Methods                                   |       |
| Normalized View                                |     |          |                                           | 494   |
| Viewport Clipping                              |     |          |                                           |       |
| Clipping in Homogeneous Coordinates            |     |          | Light Sources                             |       |
| Hardware Implementations                       |     |          | Basic lllumination Models Ambient Light   |       |
| Three-Dimensional Viewing                      |     |          | Diffuse Reflection                        |       |
| Functions                                      |     |          | Specular Reflection                       |       |
| Summary                                        |     |          | and the Phong Model                       |       |
| References                                     |     |          | Combined Diffuse and Specular             |       |
| Exercises                                      |     |          | Reflections with Multiple Light           |       |
|                                                |     |          | Sources                                   |       |
|                                                |     |          | Warn Model                                |       |
| Visi ble-Su dace Detection                     |     |          | Intensity Attenuation                     |       |
| Methods                                        | 469 |          | Color Considerations                      |       |
|                                                |     |          | Transparency                              |       |
| Classification of Visible-Surface              |     |          | Shadows                                   |       |
| D~tection Algorithms                           |     |          | Displaying Light Intensities              |       |
| Back-Face Detection                            |     |          | Assigning Intensity Levels                |       |
| Depth-BufferMethod                             |     |          | Gamma Correction and Video                |       |
| A-Buffer Method                                |     |          | Lookup Tables                             |       |
| Scan-LineMethod                                |     |          | Displaying Continuous-Tone                |       |
| Depth-Sorting Method                           |     |          | Images                                    |       |
| BSP-TreeMethod                                 |     |          | HalftonePatterns and Dithering Techniques |       |
| Area-SubdivisionMethod                         |     |          | Halftone Approximations                   |       |
| Octree Methods                                 |     |          | Dithering Techniques                      |       |
| Ray-CastingMethod                              |     |          | Polygon-RenderingMethods                  |       |
| Curved Surfaces                                |     |          | Constant-Intensity Shading                |       |
| Curved-Surface Representations                 |     |          | Gouraud Shading                           |       |
| Surface Contour Plots                          |     |          | Phong Shading                             |       |

14-6

14-7

14-8

14-9

15

15-1

15-2

15-3

15-4

15-5

xiV

Contents

Fast Phong Shading

Ray-Tracing Methods

Basic Ray-Tracing Algorithm

Contents

Calculations

527

526

528

531

15-6

15-7

15-8

15-9

CMY Color Model

HSV Color Model

Conversion Between HSV

and RGB Models

HLS Color Model

|           | Fast Phong Shading                                            |         | 15-6   | CMY Color Model                                      |     |
|-----------|---------------------------------------------------------------|---------|--------|------------------------------------------------------|-----|
|           | Ray-Tracing Methods                                           |         | 15-7   | HSV Color Model                                      |     |
|           | Basic Ray-Tracing Algorithm                                   |         | 15-8   | Conversion Between HSV and RGB Models                |     |
|           | Ray-SurfaceIntersection CaIculations                          |         | 15-9   | HLS Color Model                                      |     |
|           | Reducing Object-Intersection Calculations                     |         | 1 5-10 | Color Selection and Applications                     |     |
|           | Space-Subdivision Methods                                     |         |        | Summary                                              |     |
|           | AntiaIiased Ray Tracing                                       |         |        | Reierences                                           |     |
|           | Distributed Ray Tracing                                       |         |        |                                                      |     |
|           | Radiosity Lighting Model                                      |         |        | Exercises                                            |     |
|           | Basic Radiosity Model Progressive Refinement Radiosity Method |         | 16     | Computer Animation                                   | 583 |
|           | Environment Mapping                                           |         |        |                                                      |     |
|           | Modeling Surface Detail with Polygons                         |         | 16-2   | General Computer-Animation Functions                 |     |
|           | Texture Mapping                                               |         | 16-3   | Raster Animations                                    |     |
|           | Procedural Texturing Methods                                  |         | 16-4   | Computer-Animation Languages                         |     |
|           | Bump Mapping                                                  |         | 16-5   | Key-Frame Systems                                    |     |
|           | Frame Mapping                                                 |         |        | Morphing Simulating Accelerations                    |     |
|           | Summary                                                       |         | 16-6   | Motion Specifications                                |     |
|           | References                                                    |         |        | Direct Motion Specification                          |     |
|           |                                                               |         |        | Goal-DirectedSystems                                 |     |
|           | Exercises                                                     |         |        | Kinematicsand Dynamics                               |     |
|           | Color Models and . ,                                          | Color   |        | Summary                                              |     |
|           | Apd ications                                                  | 564     |        | References Exercises                                 | 597 |
| 15-1      | Propertieso f Light Standard Primariesand the                 | 565     | A      | Mathematics for Computer                             | 599 |
| 15-2      | Chromaticity Diagram XYZ Color Model                          | 568 569 |        | Graphics                                             |     |
|           | CIE Chromaticity Diagram                                      | 569     | A-1    | Coordinate-ReferenceFrames Two-Dimensional Cartesian | 600 |
| 15-3 15-4 | Intuitive Color Concepts RGB Color Model                      | 571 572 |        | Reference Frames                                     | 600 |
| 15-5      |                                                               |         |        |                                                      |     |
|           | YIQ Color Model                                               | 574     |        |                                                      | 601 |
|           |                                                               |         |        | Polar Coordinates in the xy Plane                    |     |

574

575

578

579

A-2

A-3

A-4

Three-Dimensional Cartesian

Reference Frames

Three-Dimensional Curvilinear

Coordinate Systems

Solid Angle

Matrices

602

602

604

A-5

A-6

Contents

Matrix Transpose

Determinant of a Matrix

Matrix Inverse

Complex Numbers

613

613

614

615

617

|     | Three-Dimensional Cartesian               |     |              | Matrix Transpose                                         |
|-----|-------------------------------------------|-----|--------------|----------------------------------------------------------|
|     | Reference Frames                          |     |              | Determinant of a Matrix                                  |
|     | Three-Dimensional Curvilinear             |     |              | Matrix Inverse                                           |
|     | Coordinate Systems                        |     |              | Complex Numbers                                          |
|     | Solid Angle                               |     |              | Quaternions                                              |
| A-2 | Points and Vectors                        |     |              |                                                          |
|     | Vector Addition and Scalar                |     |              | Nonparametric Representations Parametric Representations |
|     | Multiplication                            |     |              | Numerical Methods                                        |
|     | Scalar Product of TwoVectors              |     |              | Solving Setsof Linear Equations                          |
|     | Vector Product of TwoVectors              |     |              | Finding Roots of Nonlinear                               |
| A-3 | Basis Vectors and the Metric Tensor       |     |              | Equations                                                |
|     | Orthonormal Basis Metric Tensor           |     |              | Evaluating Integrals                                     |
| A-4 | Matrices                                  |     |              | FittingCUN~S to Data Sets                                |
|     | Scalar Multiplication and Matrix Addition | 612 | BIBLIOGRAPHY |                                                          |
|     | Matrix Multiplication                     | 612 | INDEX        |                                                          |

XV

## Contents

## Graphics C Version Computer Graphics## Graphics C Version Computer Graphics

CHAPTER -

A Survey of Computer

Graphics

<!-- image -->

<!-- image -->

omputers have become a powerful tool for the rapid and economical pro- duction of pictures. There is virtually no area in which graphical displays

Today, we find computer graphics used routinely in such diverse areas as science,

C omputers have become a powerful tool for the rapid and economical production of  pictures. There is virtually no area in which graphical displays cannot be used to some advantage, and so it is not surprising to find the use of computer  graphics so widespread. Although early applications in engineering and science had  to rely on expensive and cumbersome equipment, advances in computer technology have made interactive computer graphics a practical tool. Today, we find computer graphics used routinely in such diverse areas as science, engineering, medicine, business, industry, government, art, entertainment, advertising, education, and training. Figure 1-1  summarizes the many applications of   graphics in  simulations, education, and  graph  presentations. Before we get into  the  details  of  how  to  do computer  graphics, we  first  take  a  short  tour through a gallery of graphics applications.

400

100

200

J

100

Figure 1-i

Corporation.)

<!-- image -->

-

F ' I ~ ~ I I ~ 1 -I Examples of  computer graphics applications.  (Courtesy  o f DICOMED Corpora! ion.)

A Survey of Computer Graphics

## COMPUTER-AIDED DESIGN

A major use of   computer graphics is in design processes, particularly for engineering and architectural systems, but almost all products are now computer designed. Generally referred to as CAD, computer-aided design methods are now routinely used in the design of buildings, automobiles, aircraft, watercraft, spacecraft, computers, textiles, and many, many other products.

For some design applications;  objeck are f&amp;t  displayed in a wireframe outline form that shows the overall sham and internal features of  obiects. Wireframe displays also allow designers to qui'ckly  see the effects of  interacthe adjustments to design shapes. Figures 1-2 and 1-3  give examples of  wireframe displays in design applications.

Software packages  for  CAD applications  typically  provide  the  designer with a multi-window environment, as in Figs. 1-4 and 1-5. The various displayed windows can show enlarged sections or different views of  objects.

Circuits such as the one shown in Fig. 1-5 and  networks for comrnunications, water supply, or other utilities aR constructed with repeated placement of a few graphical shapes. The shapes used in a design represent the different network or circuit components. Standard shapes for electrical, electronic, and logic circuits are often supplied  by  the design package. For other applications, a designer can create personalized symbols that are to be  used to constmct the network or circuit.  The system is then designed by successively placing components into the layout, with the graphics package automatically providing the connections between components. This allows the designer t~ quickly try out alternate circuit schematics for  minimizing the number  of  components  or the space re--quired for the system.

Figure 1-2 Color-coded wireframe display for an automobile wheel assembly. (Courtesy o f   Emns b Sutherland.)

<!-- image -->

<!-- image -->

¡al

<!-- image -->

(b)

Figure 1-3 Color-coded wireframe displays of body designs for an aircraft and an automobile. (Courtesy o f (a)  Ewns 6  Suthcrhnd and ( b ) Megatek Corporation.)

Animations are often used in CAD applications. Real-time animations using wiseframe displays on a video monitor are  u s e f u l for testing perfonuance o f   a veh i c l e or system, as demonstrated in Fig. ld.  When we do  not display o b j s with rendered surfaces, the calculations  for each segment of  the animation can be performed quickly to produce a smooth real-time motion on the screen. Also, wireframe displays allow the designer to see into the interior of  the vehicle and  to watch the behavior of  inner components during motion. Animations in virtualreality environments are  u s e d to determine how vehicle operators are affected  by

Figure 1-4 Multiple-window, color-coded CAD workstation displays. (Courtesy o f   Intergraph Corporation.)

<!-- image -->

<!-- image -->

<!-- image -->

Figure 1-5 A drcuitdesign application,  using multiple windows and colorcded l o g i c   components, displayed on a Sun workstation with attached speaker and microphone. (Courtesy of  Sun Microsystems.)

<!-- image -->

- .

-

Figure 1-6 Simulation o f   vehicle performance during lane changes. (Courtesy  o f Ewns 6 Sutherland and Mechanical Dynrrrnics, lnc.)

certain motions. As the tractor operator in Fig. 1-7 manipulates the controls, the headset presents a stereoscopic  view (Fig.  1-8) of  the front-loader bucket or t h e backhoe, just  as if  the operator w e r e in the tractor seat. This allows the designer to explore various positions of  the bucket or backhoe that might obstruct the o p erator's  view, which can then be taken into account in the overall hactor design. Fig u r e 1-9 shows a composite, wide-angle  view from the tractor seat, displayed on a standard video monitor instead of  in a virtual threedimensional scene. And Fig. 1-10  shows a view o f   the tractor that can be displayed in a separate window or  on another monitor.<!-- image -->

Figure 1-5 A drcuitdesign application,  using multiple windows and colorcded l o g i c   components, displayed on a Sun workstation with attached speaker and microphone. (Courtesy of  Sun Microsystems.)

<!-- image -->

- .

-

Figure 1-6 Simulation o f   vehicle performance during lane changes. (Courtesy  o f Ewns 6 Sutherland and Mechanical Dynrrrnics, lnc.)

certain motions. As the tractor operator in Fig. 1-7 manipulates the controls, the headset presents a stereoscopic  view (Fig.  1-8) of  the front-loader bucket or t h e backhoe, just  as if  the operator w e r e in the tractor seat. This allows the designer to explore various positions of  the bucket or backhoe that might obstruct the o p erator's  view, which can then be taken into account in the overall hactor design. Fig u r e 1-9 shows a composite, wide-angle  view from the tractor seat, displayed on a standard video monitor instead of  in a virtual threedimensional scene. And Fig. 1-10  shows a view o f   the tractor that can be displayed in a separate window or  on another monitor.

Figure 1-7

moved, the operator views the front loader, backhoe, and surroundings

Operating a tractor in a virtual-reality environment. As the controls are through the headset. (Courtesy of the National Center for Supercomputing

Inc.)

<!-- image -->

-  -

-

-  - -

Figure 1-7 the contFols  are surroundings

Operating a tractor I n a virtual-dty  envimnment. As moved, the operator views the front lo a d e r , backhoe, and through the headset. (Courtesy o f   the National Center for Supercomputing Applicath, Univmity of Illinois at Urba~Chrrmpign, and Catopillnr, Inc.)

Figure 1-8

A headset view of the backhoe presented to the tractor operator.

(Courtesy of the National Center for

<!-- image -->

Supercomputing Applications,

Figure 1-8 A headset view of the backhoe presented to the tractor operator. (Courtesy o f   the Notional Centerfor Supcomputing Applications, UniwrsifV  of  Illinois at  Urbam~hrrmpi&amp;nd Caterpillnr,  Inc.)

Operator's view of the tractor bucket, composited in several

<!-- image -->

on a standard monitor. (Courtesy of sections to form a wide-angle view

Figure 1-9 Operator's  view of the tractor bucket, cornposited in several sections to form  a  wide-angle view on a standard monitor. (Courtesy oi the National Centerfor Supercomputing Applications, University of  lllinois at  UrhnoChmpign,  and Caterpillnr,  Inc.)

7

A Survey of Computer Graphics

Figure 1-10 View o f   the tractor displayed on a standad monitor. (Courtesy of t k National Cmter  for  Superwmputing ApplicPths, Uniwrsity  of Illinois a t UrbP~Uwmpign, and Gterpilhr, Inc.)

<!-- image -->

When  obpd designs are complete,  or nearly  complete,  realistic  lighting models and surface rendering are applied to produce displays that wiU  show the appearance of  the final product. Examples of   this are given in Fig. 1-11. Realistic displays  are a l s o generated  for advertising  o f   automobiles and other vehicles using special  lighting e f f e c t s and background scenes (Fig. 1-12).

The manufaduring process is also tied in to the computer description o f   d e signed objects to automate the construction of  the product. A circuit board layout,  for  example,  can  be transformed  into  a  description  of the  individud processes needed to construct the layout. Some mechanical parts are manufactured by describing how the surfaces are to be formed with machine tools. Figure 1-13 shows the path to be  taken by  machine tools over the surfaces of  an object during its construction. Numerically controlled machine tools are then set up to manufacture the part according to these construction layouts.

<!-- image -->

Figure 1-11

~ealistic renderings  o f   design products. (Courtesy of  fa)  Intergraph Corpomtion and fb) Emns b Sutherland.)

Studio lighting effects and realistic

<!-- image -->

o produce advertisin urface rendering everies at

pieces for finished products. The data for this rendering of a Chrysler

Figure 1-12 Studio lighting effects  and realistic surfacerendering techniques are applied to produce advertising pieces for finished products. The data for this rendering  of  a Chrysler La s e r was supplied by Chrysler Corporation. (Courtesy o f Eric Haines, 3DIEYE Inc. )

GARAGE

Figure 1-14

Visuals, Inc., Boulder, Colorado.)

A CAD layout for describing the

<!-- image -->

of a part. The part surface is numerically controlled machining

displayed in one color and the tool

Figure 1-13 A CAD layout for describing the numerically controlled machining of  a part. The part surface is displayed in one mlor and the tool path in another color. (Courtesy  o f Los Alamm National Labomtoty.)

NOOK

<!-- image -->

9

Figure 1-14 Architectural CAD layout for a building  design. (Courtesy  of  Precision Visuals, Inc., Boulder, Colorado.)

Architectural CAD layout for a building design. (Courtesy of Precision

DINING ROOM

Chapter 1

A Survey of Computer Graphics

Architects use interactive graphics methods to lay  out floor plans, such as Fig. 1-14, that show the positioning of  rooms,  doon, windows, stairs, shelves, counters, and other building features.  Working from the display  o f   a building layout on a  video monitor, an electrical designer can try  out  arrangements  for wiring, electrical outlets, and fire warning systems. Also, facility-layout packages can be applied  to the layout to determine  space utilization in  an office or on  a manufacturing floor.

Realistic displays o f   architectural  designs, as in Fig. 1-15, permit both architects and their clients to study the appearance of  a single building or a group of buildings, such as a campus or industrial complex. With virtual-reality systems, designers can even go for a simulated "walk" through the rooms or around the outsides of  buildings to  better appreciate the overall effect of a particular design. In  addition  to realistic exterior building  displays,  architectural CAD packages also provide facilities for experimenting with three-dimensional interior layouts and lighting (Fig. 1-16).

Many other kinds of  systems and products are designed using either general CAD packages or specially dweloped CAD software. Figure 1-17, for example, shows a rug pattern designed with a CAD system.

<!-- image -->

ia:

<!-- image -->

-

Figrrre 1-15 Realistic,  three-dimensional  rmderings  o f   building designs. (a) A street-level perspective for the World Trade Center project. (Courtesy of  Skidmore, Owings &amp; Mmill.) (b) Architectural  visualization o f   an  atrium, created for a compdter animation  by Marialine Prieur, Lyon, France. (Courtesy of  Thomson Digital Imngc, Inc.)

## . -

## PRESENTATION GRAPHICS

Another major applicatidn ama is presentation graphics, used to produce illustrations for reports or to generate 35-mm slides or transparencies for use with projectors. Presentation graphics is commonly used to summarize financial, statistical, mathematical, scientific, and economic data for research reports, manage rial reports, consumer information bulletins, and other types of  reports. Workstation devices and service bureaus exist for converting screen displays into 35-mm slides or overhead transparencies for use in presentations. Typical examples of presentation graphics are bar charts, line graphs, surface graphs, pie charts, and other displays showing relationships  between multiple parametem.

Figure 1-18 gives examples of  two-dimensional graphics combined with g e ographical information. This illustration shows three colorcoded bar charts combined  onto one graph and a  pie chart with three sections. Similar graphs and charts can be displayed  in three dimensions to provide additional information. Three-dimensional  graphs are sometime used simply for  effect; they can provide a more dramatic or more attractive presentation of data relationships. The charts in Fig. 1-19 include a three-dimensional bar graph and an exploded pie chart.

Additional examples of  three-dimensional graphs are shown in Figs. 1-20 and 1-21. Figure 1-20 shows one kind of surface plot, and Fig. 1-21 shows a twodimensional contour plot with a height surface.

Figtin 1-16 A hotel corridor providing a sense o f   movement by placing light fixtures along an undulating  path and creating a sense o f enhy by using light towers at each hotel room. (Courtesy of  Skidmore, Owings B Menill.)

<!-- image -->

Figure 1-17 Oriental rug pattern created with computer graphics design methods. (Courtesy of  Lexidnta Corporation.)

<!-- image -->## . -

## PRESENTATION GRAPHICS

Another major applicatidn ama is presentation graphics, used to produce illustrations for reports or to generate 35-mm slides or transparencies for use with projectors. Presentation graphics is commonly used to summarize financial, statistical, mathematical, scientific, and economic data for research reports, manage rial reports, consumer information bulletins, and other types of  reports. Workstation devices and service bureaus exist for converting screen displays into 35-mm slides or overhead transparencies for use in presentations. Typical examples of presentation graphics are bar charts, line graphs, surface graphs, pie charts, and other displays showing relationships  between multiple parametem.

Figure 1-18 gives examples of  two-dimensional graphics combined with g e ographical information. This illustration shows three colorcoded bar charts combined  onto one graph and a  pie chart with three sections. Similar graphs and charts can be displayed  in three dimensions to provide additional information. Three-dimensional  graphs are sometime used simply for  effect; they can provide a more dramatic or more attractive presentation of data relationships. The charts in Fig. 1-19 include a three-dimensional bar graph and an exploded pie chart.

Additional examples of  three-dimensional graphs are shown in Figs. 1-20 and 1-21. Figure 1-20 shows one kind of surface plot, and Fig. 1-21 shows a twodimensional contour plot with a height surface.

Figtin 1-16 A hotel corridor providing a sense o f   movement by placing light fixtures along an undulating  path and creating a sense o f enhy by using light towers at each hotel room. (Courtesy of  Skidmore, Owings B Menill.)

<!-- image -->

Figure 1-17 Oriental rug pattern created with computer graphics design methods. (Courtesy of  Lexidnta Corporation.)

<!-- image -->

12

Chapter 1

A Survey of Computer Graphics

DICT OF CIL TANKER SHILL IN CHISAPEAKE BOY

## A  SUN^^ of Computer Graph~s

Figure 1-18

(Courtesy of Computer Associates,

<!-- image -->

copyright ©1992. All rights reserved.)

Figure 1-18 Two-dimensional bar chart and me chart hked  to a geographical c l h . (Court~sy of Computer Assocbtes, copyrighi 0  1992: All  rights reserved.)

<!-- image -->

rights reserved.)

Figure 1-20 Showing relationships  with a surface  chart. (Courtesy o f   Computer Associates, copyright O 1992. All rights reserved.)

Figure 1-19

(Courtesy of Computer Associates,

<!-- image -->

copyright © 1992. All rights reserved.)

Figure 1-19 Three-dimensional bar chart. exploded pie chart, and line graph. (Courtesy of Cmnputer Associates, copyi'ghi 6 1992: All rights reserved.)

T EX-MA

<!-- image -->

ground plane. (Courtesy of Computer

Associates, copyright © 1992. All

Figure 1-21 Plotting two-dimensional  contours in the &amp;und plane, w i t h a height field plotted as a surface  above the pund plane. (Cmrtesy of   Computer Associates, copyright 0  1992. All rights d . j

Figure 1-22

Time chart displaying relevant information about project tasks.

<!-- image -->

(Courtesy of Computer Associates, copyright ©1992. All rights reserved.)

the progress of projects.

1-3

Figure 1-22 Ti e   chart displaying  relevant information about ppct tasks. (Courtesy of c o m p u t e r Associntes, copyright 0 1992.  ,411 rights m d . )

Figure 1-22 illustrates a time chart used in task planning. Tine ch a r t s and task  network layouts are used  in project  management to schedule and monitor the progess o f   propcts.

Computer graphics methods are widely used in both fine art and commercial ar

## 1-3

## COMPUTER ART

Computer graphics methods are widely used in both fine art and commercial art applications.  Artists use a variety o f   computer methods, including special-purp&amp;e hardware, artist's paintbrush (such as Lumens), other paint packages (such as Pixelpaint and Superpaint), specially developed software, symbolic mathematits packages (such as Mathematics), CAD paclpges, desktop publishing software, and animation packages that provide faciliHes for desigrung object shapes and specifiying  object motions.

Figure 1-23 illustrates the basic idea behind  a paintbrush program that allows artists to "paint" pictures  on the screen o f   a video monitor. Actually, the picture is usually painted electronically on a graphics tablet (digitizer) using a stylus,  which can simulate different brush  strokes, brush  widths,  and  colors. A paintbrush program was used to m t e   the characters in Fig. 1-24, who seem to be busy on a creation of their own.

A paintbrush system, with a Wacom cordlek, pressure-sensitive stylus, was used to  produce the electronic pa i n t i n g in Fig.  1-25 that  simulates the brush strokes of  Van Gogh. The stylus transIates changing hand presswe into variable line widths, brush  sizes, and color gradations. Figure 1-26 shows a watercolor painting produced with this stylus  and with software  that allows the artist to create watercolor, pastel, or oil brush effects that simulate  different drying out times, wetness, and  footprint. Figure 1-27 gives an  example of  paintbrush  methods combined with scanned images.

Fine artists use a variety o f   other computer technologies  to produce images. To  create pictures such as the one shown in Fig. 1-28, the artist uses a combination  of  three-dimensional modeling packages,  texture mapping, drawing  programs, and CAD software. In Fig. 1-29, we have a painting produced on a pen

Section 1-3

Computer Art

## kclion 1-3

Computer Art

13

14

Figure 1-23

Imaging.)

Figure 1-23 Cartoon drawing produced with a paintbrush  program, symbolically illustrating an artist at work on a video monitor. (Courtesy of Gould Inc., Imaging 6 Graphics Division and Aurora Imaging.)

<!-- image -->

plotter  with specially designed software that can m a t e  "automatic art"  without intervention from the artist.

Figure 1-30  shows an example of  "mathematical" art. This artist uses a cornbiation o f mathematical  fundions,  fractal  procedures,  Mathematics software, ink-jet printers,  and  other systems to create a  variety of  three-dimensional and two-dimensional  shapes and stereoscopic image pairs. Another example o f   elm-

Cartoon demonstrations of an "artist" creating a picture with a paintbrush system. The picture, drawn on a graphics tablet, is displayed on the video monitor as the elves look on. In (b), the cartoon is superimposed

camera, then scaled and positioned. (Courtesy Gould Inc., Imaging &amp; Graphics Division and Aurora Imaging.)

<!-- image -->

on the famous Thomas Nast drawing of Saint Nicholas, which was input to the system with a video

Figure 1-24

Cartoon demonstrations  o f   an "artist" mating a picture with a paintbrush  system. The picture, drawn on a graphics tablet, is displayed on the video monitor as the elves look on. In (b), the cartoon is superimposed on the famous  Thomas Nast drawing o f   Saint Nicholas, which was input to the system with a video camera, then scaled and positioned. (Courtesy Gould Inc., Imaging &amp; Gmphics Division and Aurora Imaging.)

<!-- image -->

Figure 1-25

A Van Gogh look-alike created by

<!-- image -->

An electronic watercolor, painted by John Derry of Time Arts, Inc.

with a cordless, pressure-sensitive graphics artist Elizabeth O'Rourke

stylus. (Courtesy of Wacom

<!-- image -->

using a cordless, pressure-sensitive stylus and Lumena gouache-brush

Figure 1-25 A Van Gogh look-alike created by graphcs artist E&amp;abeth O'Rourke with a cordless, pressuresensitive stylus. (Courtesy o f   Wacom Technology Corpomtion.)

Figure 1-27

about our entanglement with technology using a personal computer

Figure 1-26 An elechPnic watercolor, painted by John Derry of   Tune Arts, Inc. using a cordless,  pressure-sensitive stylus and Lwnena gouache-brush &amp;ware. (Courtesy of Wacom Technology  Corporation.)

The artist of this picture, called Electronic Avalanche, makes a statement with a graphics tablet and Lumena software to combine renderings of

<!-- image -->

## Figure 1-27

The artist of   this picture, called Electrunic Awlnnche, makes a statement about our entanglement with technology using a personal computer with a graphics tablet and Lumena software  to combine renderings o f leaves, Bower petals, and electronics componenb  with scanned images. (Courtesy  of  the Williams Gallery.  w g h t 0  1991 by Imn Tnrckenbrod, Tke School o f   the Arf Instituie o f   Chicago.)

15

Figure 1-28

(entitled, Whigmalaree) was created with a combination of

<!-- image -->

Figure 1-29

<!-- image -->

Electronic art output to a pen methods using a graphics tablet, three-dimensional modeling,

plotter from software specially texture mapping, and a series of transformations. (Courtesy of the

designed by the artist to emulate his style. The pen plotter includes

Figwe 1-28 From a series called Sphnrs o f Inpumce, this electronic painting (entitled, WhigmLaree) was awted with a combination o f methods  using a graphics  tablet, three-dimensional modeling, texture mapping, and a s e r i e s o f   transformations.  (Courtesy  of the Williams Gallery. Copyn'sht (b 1992 by w n e   RPgland,]r.)

Figure 1-30

Figure 1-29 Electronic art output to a pen plotter from software specially designed by the artist to emulate hi s style. The pen plotter includes multiple pens and painting inshuments, including Chinese brushes. (Courtesy  o f   the Williams Gallery. Copyright 8 by Roman Vmtko, Minneapolis College o f Art 6 Design.)

Figure 1-31

fractal procedures, and

<!-- image -->

Department of Computer Science, Indiana University. The image

Using mathematical functions,

<!-- image -->

was rendered using Mathematica and Wavefront software.

16

supercomputers, this artist- composer experiments with various

Figure 1-30 This creation is  based on a visualization  of Fermat's Last Theorem, I" + y" = z" , with n = 5, by Andrew Hanson, Department of Computer Science, Indiana University. The image was rendered using Mathematics and Wavefront sof t w a r e . (Courtesy  o f   the Williams Gallery. Copyright 8 1991 by Stcmrt Dirkson.)

Figure 1-31 U s i n g mathematical hlnctiow, fractal  procedures,  and supermmpu ters, this artistcomposer experiments  with various designs  to synthesii form and color with musical composition. (Courtesy of Brian Ewns, Vanderbilt University.)Figure 1-28

(entitled, Whigmalaree) was created with a combination of

<!-- image -->

Figure 1-29

<!-- image -->

Electronic art output to a pen methods using a graphics tablet, three-dimensional modeling,

plotter from software specially texture mapping, and a series of transformations. (Courtesy of the

designed by the artist to emulate his style. The pen plotter includes

Figwe 1-28 From a series called Sphnrs o f Inpumce, this electronic painting (entitled, WhigmLaree) was awted with a combination o f methods  using a graphics  tablet, three-dimensional modeling, texture mapping, and a s e r i e s o f   transformations.  (Courtesy  of the Williams Gallery. Copyn'sht (b 1992 by w n e   RPgland,]r.)

Figure 1-30

Figure 1-29 Electronic art output to a pen plotter from software specially designed by the artist to emulate hi s style. The pen plotter includes multiple pens and painting inshuments, including Chinese brushes. (Courtesy  o f   the Williams Gallery. Copyright 8 by Roman Vmtko, Minneapolis College o f Art 6 Design.)

Figure 1-31

fractal procedures, and

<!-- image -->

Department of Computer Science, Indiana University. The image

Using mathematical functions,

<!-- image -->

was rendered using Mathematica and Wavefront software.

16

supercomputers, this artist- composer experiments with various

Figure 1-30 This creation is  based on a visualization  of Fermat's Last Theorem, I" + y" = z" , with n = 5, by Andrew Hanson, Department of Computer Science, Indiana University. The image was rendered using Mathematics and Wavefront sof t w a r e . (Courtesy  o f   the Williams Gallery. Copyright 8 1991 by Stcmrt Dirkson.)

Figure 1-31 U s i n g mathematical hlnctiow, fractal  procedures,  and supermmpu ters, this artistcomposer experiments  with various designs  to synthesii form and color with musical composition. (Courtesy of Brian Ewns, Vanderbilt University.)

tronic art created with the aid of mathematical relationships is shown in Fig. 1-31. Section 1-3

The artwork of this composer is often designed in relation to frequency varia- tions and other parameters in a musical composition to produce a video that inte

grates visual and aural patterns.

erating electronic images in the fine arts, these methods are also applied in com-

Although we have spent some time discussing current techniques for gen- tronic art created with the aid of  mathematical relationships  is shown in Fig. 1-31. The artwork of  this composer is often designed  in  relation to frequency variations and other parameters in a musical composition to produce a video that integrates visual and aural patterns.

Although we have spent some time discussing current techniques for generating electronic images in the fine arts, these methods are a l s o applied in commercial art for logos and other designs, page layouts combining text and graphics, TV advertising  spots, and other  areas. A workstation  for producing  page layouts that combine text and graphics is ihstrated in Fig. 1-32.

For many applications of  commercial art (and in motion pictures and other applications), photorealistic techniques are used to render images of  a product. Figure 1-33 shows an example of logo design, and Fig. 1-34 gives three computer graphics images for product advertising. Animations are also usxi frequently in advertising,  and  television  commercials are  produced  frame by  frame,  where

Figure 1-32

of Visual Technology.)

Page-layout workstation. (Courtesy

l i p r t . 1-32 Page-layout  workstation. (Courtesy oj Visunl Technology.)

<!-- image -->

Figure 1-34

Three dimensional rendering for a

Figure 1-33

logo. (Courtesy of Vertigo Technology,

Figure 1-33 Three-dimens~onal rendering for a logo. (Courtesy o f   Vertigo Technology, Inc.)

<!-- image -->

Product advertising. (Courtesy of (a) Audrey Fleisher and (b) and (c) SOFTIMAGE, Inc.)

<!-- image -->

Fi&lt;yuru 1 -34 Product advertising.

<!-- image -->

- .

(Courtesy  oj la)  Audrey Fleisherand lb)  and lc)  SOFTIMAGE,  Inc.)

<!-- image -->

17

## seaion 1-3

Computer Art

Computer Art

18

Chapter 1

A Survey of Computer Graphics each frame of the motion is rendered and saved as an image file. In each succes-

sive frame, the motion is simulated by moving object positions slightly from their been rendered, the frames are transferred to film or stored in a video buffer for

positions in the previous frame. When all frames in the animation sequence have playback. Film animations require 24 frames for each second in the animation se

## Chapter 1

A Survey of Computer Graphics each frame of  the motion is rendered and saved as an image file. In each successive  frame, the motion is simulated  by moving o w positions slightly from their positions in  the previous frame. When all frames in the animation sequence have been mdered, the frames are transfed to film or stored in a video buffer for playback. Film animations  require 24 frames for each second in the animation sequence. I f   the animation is to be played back on a video monitor, 30 frames per second are required.

1-4

A common graphics method employed in many commercials is rnorphing, where one obiect is  transformed (metamomhosed)  into another. This method has been used in h commercials  to an oii can into an automobile  engine, an automobile into a tiger, a puddle o f   water into a t k , and one person's face into another face. An example o f   rnorphing is given in Fig. 1-40.

tures, music videos, and television shows. Sometimes the graphics scenes are dis-

## 1-4 ENTERTAINMENT

Computer graphics methods am now  commonly used in  making motion pictures, music videos, and television shows.  Sometimes  the graphics scenes are displayed by themselves, and sometimes graphics objects are combined with the actors and live scenes.

A graphics scene generated for the movie Star  Trek-% Wrath o f Khan i s shown in Fig. 1-35.  The planet and spaceship are drawn in wirefame form and will be shaded with  rendering methods to produce solid surfaces. Figure 1-36 shows scenes generated with advanced modeling and surfacerendering methods for two awardwinning short h.

Many TV  series regularly employ computer graphics methods. Figure 1-37 shows a scene pduced for the seriff Deep  Space Nine. And  Fig. 1-38 shows a wireframe person combined with actors in a live scene for the series Stay lhned.

Figure 1-35

Graphics developed for the

Trek-The Wrath of Khan. (Courtesy of

<!-- image -->

Evans &amp; Sutherland.)

~ i a ~ h i a developed for the Paramount Pi c t u r e s movie Stnr T r e k - T h e Wllrrh of Khan. (Courtesy of Ewns &amp; Sutherland.)

In Fig. 1-39, we have a highly realistic image taken from a reconstruction of   thirteenth-century Dadu (now Beijing) for a Japanese broadcast.

Music videos use graphin in  several ways. Graphics o b j e c t s can be combined with the live action, as in Fig.1-38, or graphics and image processing techniques can be used to produce a transformation o f   one person or object into another (morphing). An example of morphing is shown in the sequence of scenes in Fig. 1-40, produced for the David Byme video She's Mad.

<!-- image -->

(a)

Fiprc 1-36 (a) A computer-generated  scene from the film M s Dmm, copyright O Pixar 1987. (b) A computer-generated scene from the film K n i c M , copyright O Pixar 1989. (Courfesy  of

Pixar.)

<!-- image -->

-

-  - -

I

i p r c

1-

. -

17

A graphics scene in the TV series Dwp Space Nine. (Courtesy of Rhyt h m b. Hues Studios.)

- .

-  -

-

<!-- image -->

(b)

## Mi o n   1-4

Enterlainrnent

Chapter 1

A Survey of Computer Graphics

## A Survey of Computer Graphics

Figure 1-38

Figurp 1-38 Graphics combined with a L i v e scene in the TV series S t a y 7bned. (Courtesy o f   Rhythm 6 Hues St u d i o s . )

<!-- image -->

Figure 1-39

<!-- image -->

Corporation (Tokyo) and rendered with TDI software. (Courtesy of

Figure 1-39 An image from a &amp;owhuction  o f thirteenth-centwy Dadu (Beijmg today),  created by T a i s e i Corporation  (Tokyo) and rendered with TDI software. ( C o u r t e s y of Thompson D i g i t a l Image, lnc.)

20

## St*ion 1-5

## Education and Training

<!-- image -->

## 1-5

## EDUCATION AND TRAINING

Computer-generated  models  of  physical,  financial, and  economic systems are often used  as educational aids. Models of  physical systems, physiological systems, population trends, or equipment, such as the colorcoded diagram in Fig. 141, can help trainees to understand the operation of the system.

For  some training applications, special systems are designed. Examples of such specialized systems are the simulators for practice sessions or training of ship captains, aircraft pilots,  heavy-equipment operators, and air  trafficcontrol personnel. Some simulators have no video screens; for example, a flight simulator with only a control panel for instrument fly i n g . But most simulators provide graphics screens for visual operation. Two examples of  large simulators with internal viewing systems are shown in Figs. 1-42 and 1-43. Another type o f   viewing system  is  shown in  Fig. 1 4 4 . Here a  viewing screen with  multiple  panels is mounted in front of the simulator. and color projectors display the flight m e on the screen panels. Similar viewing systems are  used in simulators for training aircraft control-tower personnel. Figure 1-45 gives an example of  the inshuctor's area in a flight simulator. The keyboard is used to input parameters affeding the airplane performance or the environment, and the pen plotter is used to  chart the path of  the aircraft during a training session.

Scenes generated for various simulators are shown in Fi g s . 1-46 through 148. An output from an automobile-driving simulator is given in Fig.  1-49.  This simulator is used to investigate the behavior o f   drivers in critical situations. The drivers' reactions  are then used as a basis for optimizing vehicle design to maximize traffic safety.## St*ion 1-5

## Education and Training

<!-- image -->

## 1-5

## EDUCATION AND TRAINING

Computer-generated  models  of  physical,  financial, and  economic systems are often used  as educational aids. Models of  physical systems, physiological systems, population trends, or equipment, such as the colorcoded diagram in Fig. 141, can help trainees to understand the operation of the system.

For  some training applications, special systems are designed. Examples of such specialized systems are the simulators for practice sessions or training of ship captains, aircraft pilots,  heavy-equipment operators, and air  trafficcontrol personnel. Some simulators have no video screens; for example, a flight simulator with only a control panel for instrument fly i n g . But most simulators provide graphics screens for visual operation. Two examples of  large simulators with internal viewing systems are shown in Figs. 1-42 and 1-43. Another type o f   viewing system  is  shown in  Fig. 1 4 4 . Here a  viewing screen with  multiple  panels is mounted in front of the simulator. and color projectors display the flight m e on the screen panels. Similar viewing systems are  used in simulators for training aircraft control-tower personnel. Figure 1-45 gives an example of  the inshuctor's area in a flight simulator. The keyboard is used to input parameters affeding the airplane performance or the environment, and the pen plotter is used to  chart the path of  the aircraft during a training session.

Scenes generated for various simulators are shown in Fi g s . 1-46 through 148. An output from an automobile-driving simulator is given in Fig.  1-49.  This simulator is used to investigate the behavior o f   drivers in critical situations. The drivers' reactions  are then used as a basis for optimizing vehicle design to maximize traffic safety.

<!-- image -->

National Laboratory.)

reactor. (Courtesy of Los Alamos

Figure 1  -4  1 Color-coded  diagram used to explain the operation of a nuclear reactor. (Courtesy of Las Almnos National  laboratory.)

Figure 1-43

Figure 1-42

with a full-color visual system and

<!-- image -->

six degrees of freedom in its motion. (Courtesy of Frasca

Figure 1-42 A Me, enclosed tlight simulator with a full-color visual syst e m and six degrees of freedom in its motion.  (Courtesy of Fmxm Intematwml.)

<!-- image -->

- --

Figure 1 4 3 A military tank simulator with a visual imagery system.  (Courtesy of Mediatech and GE Aerospace.)

22

<!-- image -->

Figure 1-44

A flight simulator with an external full-color viewing system. (Courtesy of Fresca

International.)

Figure 1-44 A fight simulator  with an external full-zulor viewing system. (Courtay a f F m InternafiomI.)

Figure 1-45

<!-- image -->

instructor to monitor flight conditions and to set airplane and

An instructor's area in a flight simulator. The equipment allows the

Figure 1-45 An instructor's area in a flight sunulator.  The equipment allows the instructor to monitor flight conditions and to set airphne and environment  parameters.  (Courtesy of Frasur Infermtionol.)

Section 1-5

Education and Training

kction 1-5

## Edwtion and Training

<!-- image -->

23

24

Chapter 1

A Survey of Computer Graphics

<!-- image -->

Figure 1-46

F i p 1-46 Flightsimulator  imagery. ((Courtesy 4  E m n s   6  Sutherfund.)

<!-- image -->

Figure 1-47

simulator. (Courtesy of Evans &amp;

<!-- image -->

Sutherland.)

-

Figure 1-47 Imagery generated f o r a naval simulator.  (Courtesy o f   Ewns 6 Sutherlrmd.)

Figure 1-48

Mediatech and GE Aerospace.)

Figlire 1-48 Space shuttle imagery.  (Courtesy of Mediatech and GE Aerospce.)

<!-- image -->

Figure 1-49

Imagery from an automobile simulator used to test driver

reaction. (Courtesy of Evans &amp;

Sutherland.)

1-6

VISUALIZATION

Figure 1-49 Imagery from an automobile simulator used to test driver reaction. (Courtesy of Evans 6 Sutherlrmd.)

<!-- image -->

to analyze large amounts of information or to study the behavior of certain

## 1-6

## VISUALIZATION

Scientists, engineers, medical personnel, business analysts, and others often need to  analyze  large  amounts  of  information or  to  study  the  behavior  of  certain processes. Numerical simulations carried out on supercomputers frequently produce data files containing thousands and even millions of  data values. Similarly, satellite cameras and other sources are amassing large data files faster than they can be interpreted. Scanning these large sets of  n u m b a   to determine trends and relationships is a tedious and ineffective process. But if  the data are converted to a visual form, the trends and patterns are often immediately apparent. Figure 150 shows an example of  a large data set that has been converted to a color-coded display o f   relative heights above a ground plane. Once we have plotted the density values in this way, we can see easily the overall pattern of  the data. Producing graphical representations for scientific, engineering, and  medical data sets and processes is generally referred to as scientific visualization. And the tenn business  v i s u a l i z a t i o n   is used in connection  with data sets related to commerce, industr y , and other nonscientific areas.

There are many  different  kinds  o f   data sets, and  effective  visualization schemes depend on the characteristics of the data. A collection of  data can contain scalar values,  vectors, higher-order tensors, or any combiytion of  these data types. And data sets can be two-dimensional  or threedimensional. Color coding is just  one way  to  visualize a  data set. Additional  techniques include contour plots, graphs and charts, surface renderings, and visualizations of  volume interiors.  In  addition,  image  processing  techniques  are  combined  with  computer graphics to produce many of  the data visualizations.

Mathematicians, physical scientists,  and others use visual techniques to analyze mathematical functions and  processes  or  simply  to  produce  interesting graphical representations. A color plot of mathematical curve functions is shown in Fig. 1-51, and a surface plot o f   a function is shown in Fig. 1-52. Fractal proce-

Section 1-6

Visualization

Visualization

25

26

Chapter 1

A Survey of Computer Graphics

## A Survey of  Computer  Graphics

Figure 1-50

<!-- image -->

-

.-

Figure 1-50 A color-coded plot with 16 million density points  of relative brightness ob~t?~ed for the Whirlpool Nebula reveals two distinct galaxies. (Courtesy of Lar A I a m National Laboratory.)

Figure 1-51

plotted in various color

Mathematical curve functions

Prueitt, Los Alamos National

<!-- image -->

Laboratory.)

Figure 1-51 Mathematical curve functiow plotted in various color combinations. (Courtesy ofMeluin L. Prun'tt, Los Alamos National Laboratory.)

Figure 1-52

rendering techniques were applied

Lighting effects and surface- to produce this surface

representation for a three-

<!-- image -->

-  -

Figurn 1-52 Li g h t i n g effects and surfacerendering techniqws were applied to produce this surface representation for a threedimensional funhon. (Courtesy o f Wf m m h m h , Inc, The h f a k e r o f Mathmurtica.)26

Chapter 1

A Survey of Computer Graphics

## A Survey of  Computer  Graphics

Figure 1-50

<!-- image -->

-

.-

Figure 1-50 A color-coded plot with 16 million density points  of relative brightness ob~t?~ed for the Whirlpool Nebula reveals two distinct galaxies. (Courtesy of Lar A I a m National Laboratory.)

Figure 1-51

plotted in various color

Mathematical curve functions

Prueitt, Los Alamos National

<!-- image -->

Laboratory.)

Figure 1-51 Mathematical curve functiow plotted in various color combinations. (Courtesy ofMeluin L. Prun'tt, Los Alamos National Laboratory.)

Figure 1-52

rendering techniques were applied

Lighting effects and surface- to produce this surface

representation for a three-

<!-- image -->

-  -

Figurn 1-52 Li g h t i n g effects and surfacerendering techniqws were applied to produce this surface representation for a threedimensional funhon. (Courtesy o f Wf m m h m h , Inc, The h f a k e r o f Mathmurtica.)

dures using quaternions  generated the object shown in Fig. 1-53,  and a topological shucture is displayed in Fig. 1-54. Scientists are a l s o developing methods for visualizing general classes of  data. Figure 1-55 shows a general  technique for graphing and modeling data distributed  over a spherical  surface.

A few o f   the many other visualization applications are shown in Figs. 1-56 through 149. T h e s e f i g k show airflow ove? ihe surface of  a space shuttle, numerical  modeling  o f thunderstorms,  study  o f aack propagation in  metals,  a colorcoded plot of  fluid density over an airfoil, a cross-sectional slicer for data sets, protein modeling, stereoscopic viewing o f   molecular structure, a model of the ocean f l o o r , a Kuwaiti oil-fire simulation, an air-pollution study, a com-growing study, rrconstruction o f Arizona's Cham CanY&amp; tuins, and a-graph  ofautomobile accident statistics.

Figure 1 -54 Four views f r o m a real-time, interactive  computer-animation study o f   minimal surface ("snails") in the 3 -   sphere projected to threedimensional Euclidean space. (Courtesy of  George Francis, Deprtmmt of  M a t h t i c s ad the Natwnal Colter  for  Sup~rromputing Applications, University o f   Illinois at UrhnaChampaign. Copyright O 1993.)

<!-- image -->

1-6 wsualization

<!-- image -->

-

Figure 1-53 A four-dimensional  object projected into threedimensional  space, then projected to a video monitor, and color coded. The obpct was generated using quaternions and fractal squaring procedm, with an Want subtracted to show the complex Julia se t . (Crmrtrsy of Iohn C.  Ifart,  School o f Electrical Enginem'ng d Computer Science, Washingfon State Uniwrsity.)

<!-- image -->

-

F+pre  1-55 A method for graphing and modeling data distributed over a spherical  surface. (Courfesy  o f   Greg Nielson. Computer Science Department,  Arizona State University.)

Chapter 1

A Survey of Computer Graphics

## A Survey of Computer Graphics

Figure 1-56

Hultquist and Eric Raible, NASA

<!-- image -->

Ames. (Courtesy of Sam Uselton,

NASA Ames Research Center.)

Figure 1-56 A vi s u a l i z a t i o n of &amp;eam surfaces flowing past a space sh u t t l e by Jeff Hdtquist and Eric Raible, NASA Ames. (Courtlsy  of Sam Wton, NASA Amcs Raaadr Cnrtlr.)

Figure 1-58

thunderstorm. (Courtesy of Bob

Atmospheric Sciences and the National

Figure 1-57

Numerical model of airflow inside a thunderstorm. (Courtesy of Bob

Atmospheric Sciences and the National

Wilhelmson, Department of

<!-- image -->

Center for Supercomputing

Figure 1-57 Numerical model of a i r f l o w   i n s i d e a thunderstorm. (Cmtrtsv of Bob

<!-- image -->

Wilhelmson, Department of

Center for Supercomputing

28

Figure 2-58 Numerical model of the surface of   a thunderstorm. (Courtsy of Sob Wilklmsbn,  Lkprhnmt of Atmospheric Sciences and t k NatiaMl Center lor Supercomputing Applications, Unimmity  ofnlinois at Urbana-Champrip.)

Figure 1-59

Color-coded visualization of stress energy density in a crack-

propagation study for metal plates, the National Center for

modeled by Bob Haber. (Courtesy of

<!-- image -->

Section 1-6

Visualization

A fluid dynamic simulation, showing a color-coded plot of fluid

density over a span of grid planes around an aircraft wing, developed

--

Champaign.)

--

Figure 1-59 Colorded  visualization o f stress energy density  in a crackpropagation  study for metal plates, modeled by Bob Haber. (Courfesy  of t k Natioml Cinter for Supercaputmg Applicutions, Unmity  of n l i ~ i s at UrbrmnChnmpa~gn.)

Figure 1-61

showing color-coded data values

Commercial slicer-dicer software, over cross-sectional slices of a data

set. (Courtesy of Spyglass, Inc.)

Figure 1-60 A fluid dynamic simulation, showing a color-coded plot o f   fluid density over a span o f   grid planes around an aircraft wing, developed by Lee-Hian Quek, John Eickerneyer, and Jeffery  Tan. (Courtesy of  the Infinnation Technology Institute, Republic of Singapore.)

Visualization of a protein structure

SDSC. (Courtesy of Stephanie Sides, by Jay Siegel and Kim Baldridge,

San Diego Supercomputer Center.)

<!-- image -->

F@w 1-61 Commercial slicer-dicer  software, showing color-coded  data values over awsedional slices o f   a  data set. (Courtesy of Spyglnss, Im.)

<!-- image -->

29

Fi p m  1-62 Visualization of a protein structure by Jay  Siege1 and Kim Baldridge, SDSC. (Courfesy  of  Stephnnie Sides, San Diego Supercomputer Cmter.)

<!-- image -->

30

Figure 1-63

of Illinois at Urbana-Champaign.)

Figure 1 -63 Stereoscopic viewing of a molecular strumup us i n g a "boom" device. (Courtesy of  the Nafiaal Centerfir Supermputing Applhtions, Univmity o f   Illinois at UrbomChnmprign.)

<!-- image -->

Figure 1-64

showing a visualization of the data, by David Sandwell and Chris

<!-- image -->

Figure 1-65

Kuwaiti oil fire, by Gary

A simulation of the effects of the

Glatzmeier, Chuck Hanson, and

Small, Scripps Institution of Ocean-

Figure  1-64 One image from a stendqnc pair, showing a visualization  of the ocean floor obtain e d from mteltik data, by David Sandwell and  C h r i s Small, Scripps Ins t i t u t i o n of   Oceanography, and Jim  Mdeod, SDSC. (Courtesy of Stephanie Sids, Sun Diego Supramrputer Center.)

Paul Hinker. (Courtesy of Mike

<!-- image -->

Krogh, Advanced Computing

Figvne 165 A simulation of   the e f f d s of t h e Kuwaiti oil f i r e , by Gary Glatpneier, Chuck Hanson,  and Paul Hinker. ((Co u r t e s y of Mike Kmzh, Adrnnced Computing lnboratwy 41 Los Alrrmos Nafionul hbomtwy.)

-

<!-- image -->

Figure 1-66 A visualization o f   pollution over the earth's surface  by Tom Palmer, Cray Research Inc./NCSC; Chris Landreth, NCSC; and Dave W, NCSC. Pollutant SO, is plotted as  a blue surface, acid-rain  deposition is a color plane on the map surface, and rain concentration  i s  shown as clear cylinders. (Courtesy of the North Cnmlim Supercomputing Center/MCNC.)

<!-- image -->

- - .

-

Figure 1-68 A visualization o f   the reconstruction o f   the ruins at Cham Canyon, Arizo n a . (Courtesy of Melvin L. Pnceitt, L o s Alamos Nationul lnboratory. Data supplied by Stephen If. Lekson.)

Section 1-6 Visualization

<!-- image -->

Figure 1-67 One frame  of an animation sequence showing the development of  a corn ear. (Couitcsy of tk National Center for Supmomputing Applimhs, Unimity  ofnlinois at UrhnaChampaign.)

Figure 1-69 A prototype technique, called WinVi, for visualizing tabular multidimensional  data is used here to correlate statistical information on pedestrians involved in automobile  accidents,  developed by a visuahzation team at I T T . (Courtesy  o f   Lee-Hian Quek, Infonnatwn Technology  Institute. Republic of  Singapore.)

<!-- image -->-

<!-- image -->

Figure 1-66 A visualization o f   pollution over the earth's surface  by Tom Palmer, Cray Research Inc./NCSC; Chris Landreth, NCSC; and Dave W, NCSC. Pollutant SO, is plotted as  a blue surface, acid-rain  deposition is a color plane on the map surface, and rain concentration  i s  shown as clear cylinders. (Courtesy of the North Cnmlim Supercomputing Center/MCNC.)

<!-- image -->

- - .

-

Figure 1-68 A visualization o f   the reconstruction o f   the ruins at Cham Canyon, Arizo n a . (Courtesy of Melvin L. Pnceitt, L o s Alamos Nationul lnboratory. Data supplied by Stephen If. Lekson.)

Section 1-6 Visualization

<!-- image -->

Figure 1-67 One frame  of an animation sequence showing the development of  a corn ear. (Couitcsy of tk National Center for Supmomputing Applimhs, Unimity  ofnlinois at UrhnaChampaign.)

Figure 1-69 A prototype technique, called WinVi, for visualizing tabular multidimensional  data is used here to correlate statistical information on pedestrians involved in automobile  accidents,  developed by a visuahzation team at I T T . (Courtesy  o f   Lee-Hian Quek, Infonnatwn Technology  Institute. Republic of  Singapore.)

<!-- image -->

Chapter 1

A Survey of Computer Graphics

32

1-7

IMAGE PROCESSING

Although methods used in computer graphics and image processing overlap, the graphics, a computer is used to create a picture. Image processing, on the other

two areas are concerned with fundamentally different operations. In computer hand, applies techniques to modity or interpret existing pictures, such as pho-

improving picture quality and (2) machine perception of visual information, as

## tographs and TV scans. Two principal applications of image processing are (1)

Although methods used in computer graphics and Image processing overlap, the amas am concerned with fundamentally different operations. In computer graphics, a computer is used to create a pichue. Image processing, on the other hand.  applies techniques to modify or interpret existing pibures,  such as p h e tographs and TV  scans. Two principal applications of  image pmcessing are (1) improving picture quality and (2) machine perception of  visual information, as used in robotics.

To apply imageprocessing methods, we first digitize a photograph or other picture into an image file. Then digital methods can be applied to rearrange picture parts, to enhance color separations, or to improve the quality o f   shading. An example of  the application o f   imageprocessing  methods to enhance the quality of  a picture is shown in Fig. 1-70. These techniques are used extensively in commercial art applications that involve the retouching and rearranging of   sections of   photographs and other artwork. Similar methods are used to analyze satellite photos of  the earth and photos of  galaxies.

Medical applications also make  extensive  use  of  imageprocessing  techniques  for picture enhancements, in  tomography and in  simulations of  operations. Tomography is  a  technique of  X-ray  photography  that allows cross-sectional  views  of  physiological  systems  to  be  displayed.  Both  computed X-rav tomography (CT) and position emission tomography (PET) use propchon methods to reconstruct  cross sections from digital data. These techniques are also used to figure 1-70

<!-- image -->

.-  -.

figure 1-70 A blurred photograph of a li c e n s e plate becomes legible  after the application o f   imageprocessing  techniques.  (Courtesy of Los  Alamos  National  Laboratory.)

monitor internal functions and show crcss sections during surgery. Other me&amp; ical imaging techniques include ultrasonics  and nudear medicine scanners. With ultrasonics, high-frequency sound waves, instead of  X-rays, are used to generate digital data. Nuclear medicine scanners colled di@tal data from radiation emitted from ingested radionuclides and plot colorcoded images.

lmage  processing  and  computer  graphics  are  typically  combined  in many applications. Medicine, for example, uses these techniques to  model and study  physical  functions, to  design  artificial  limbs,  and  to  plan  and  practice surgery. The last  application i s   generally referred  to as computer-aided surgery. Two-dimensional cross sections o f   the body are obtained using imaging techniques. Then the slices are viewed and manipulated using graphics methods to simulate actual surgical procedures and to try out different surgical cu t s .   Examples of these medical applications  are shown in Figs. 1-71  and 1-72.

Figure  1-71 One frame from a computer animation visualizing  cardiac activation levels within regions of  a semitransparent volume  rendered dog heart. Medical data provided b y Wiiam Smith, Ed Simpson, and G. Allan Johnson,  Duke University. Image-rendering software by T o m   Palmer, Cray Research, Inc./NCSC. (Courtesy of  Dave Bock, North Carolina Supercomputing CenterlMCNC.)

<!-- image -->

Figure  1-72 One  image from a stereoscopic  pair showing the bones of   a  human hand. The images were rendered b y lnmo Yoon, D. E. Thompson, and W. N. Waggempack, Jr;, LSU, from a data set obtained with CT scans by  Rehabilitation Research, GWLNHDC. These images show a possible tendon path for reconstructive surgery. (Courtesy  o f IMRLAB, Mechnniwl Engineering, Louisiow State  Uniwsity.)

<!-- image -->

~~ 1-7

Image Pm&amp;ng

.

## GRAPHICAL USER INTERFACES

It  is common  now  for  software  packages  to provide a  graphical  interface.  A major component of  a graphical interface is a window manager that allows a user to display multiple-window areas. Each window can contain a different process that  can contain  graphical  or  nongraphical  displays. To make a  particular window active, we simply click in that window using an interactive pointing dcvicc.

Interfaces also display menus and  icons for fast selection of  processing options or parameter values. An icon is a graphical symbol that is designed  to look like  the  processing  option  it  represents.  The advantages  of  icons  are  that  they take up  less screen space than corresponding textual descriptions and they can be understood  more quickly if   well designed. Menbs contain lists of textual descriptions and icons.

Figure  1-73 illustrates a  typical  graphical  mterface, containing a  window manager, menu displays, and icons. In this example, the menus allow selection of processing options,  color  values, and  graphics parameters.  The icons represent options for painting, drawing, zooming, typing text strings, and other operations connected with picture construction.

<!-- image -->

- -

Figure 1-73 A graphical user interface, showing multiple window areas, menus, and icons. (Courtmy of Image-ln Grponrtion.)

VI

CHAPTER

2

<!-- image -->

<!-- image -->

Overview of Graphics

Systems

D  ue to the widespread recognition of  the power and utility of  computer graphics in virtually all fields, a broad  range o f   graphics hardware and software systems is now  available.  Graphics capabilities for both  two-dimensional and three-dimensional  applications a x   now common on general-purpose computers, including many hand-held calculators. With personal computers, we can use a wide variety of  interactive input devices and graphics software packages. For higherquality applications, we can choose from a number of  sophisticated special-purpose  graphics hardware systems and technologies. In this chap ter, we explore the basic features  of graphics hardwa~e components and graphics software packages.

## 2-1

## VIDEO DISPLAY DEVICES

Typically, the primary output device in a graphics system is a video monitor (Fig. 2- 1 ) . The operation of most video monitors is based on the standard cathode-ray tube  (CRT) design, but several other technologies exist and solid-state monitors may eventually predominate.

<!-- image -->

-  --

-

rig~rrr 2-1 A computer graphics workstation. (Courtrsyof T h i r . Inc.)D  ue to the widespread recognition of  the power and utility of  computer graphics in virtually all fields, a broad  range o f   graphics hardware and software systems is now  available.  Graphics capabilities for both  two-dimensional and three-dimensional  applications a x   now common on general-purpose computers, including many hand-held calculators. With personal computers, we can use a wide variety of  interactive input devices and graphics software packages. For higherquality applications, we can choose from a number of  sophisticated special-purpose  graphics hardware systems and technologies. In this chap ter, we explore the basic features  of graphics hardwa~e components and graphics software packages.

## 2-1

## VIDEO DISPLAY DEVICES

Typically, the primary output device in a graphics system is a video monitor (Fig. 2- 1 ) . The operation of most video monitors is based on the standard cathode-ray tube  (CRT) design, but several other technologies exist and solid-state monitors may eventually predominate.

<!-- image -->

-  --

-

rig~rrr 2-1 A computer graphics workstation. (Courtrsyof T h i r . Inc.)

Refresh Cathode-Ray Tubes

Figure 2-2 illustrates the basic operation of a CRT. A beam of electrons (cathode that direct the beam toward specified positions on the phosphor-coated screen.

rays), emitted by an electron gun, passes through focusing and deflection systems

The phosphor then emits a small spot of light at each position contacted by the

## Refresh Cathode-Ray Tubes

Fipm 2-2  illustrates the basic operation of,a CRT. A beam of electrons (cathode rays), emitted by an electron gun, passes through focusing and deflection systems that direct the beam toward  specified  positions on the phosphomted screen. The phosphor  then emits a small spot of  light at each position contacted by  the electron beam.  Because  the  light emitted by  the  phosphor  fades very  rapidly, some method is needed for maintaining the screen picture. One way to keep the phosphor glowing is to redraw the picture repeatedly by  quickly directing the electron beam back over the same points. This type of  display is called a refresh CRT.

Section 2-1

Video Display Devices

Vkh Display Devices

The primary components of  an electron gun in a CRT  are the heated metal cathode and a control grid (Fig. 2-31. Heat is supplied to the cathode by direding a current through a coil o f   wire, called the filament, inside the cylindrical cathode structure. This causes electrons to be 'kiled  off" the hot cathode surface. In  the vacuum inside the CRT envelope, the free, negatively charged electrons are then accelerated toward the phosphor coating by a high positive voltage. The acceler-

Connector

Pins

Figure 2-2

Heating

Filament

Figure 2-3

Operation of an electron gun with an accelerating anode.

<!-- image -->

-

Figure 2-2 Basic design of   a magneticdeflection Focusing

Electron

Beam

CRT.

Path

<!-- image -->

-

CRT.

--

38

Chapter 2

Overview of Graphics Systems ating voltage can be generated with a positively charged metal coating on the in-

be used, as in Fig. 2-3. Sometimes the electron gun is built to contain the acceler- side of the CRT envelope near the phosphor screen, or an accelerating anode can

ating anode and focusing system within the same unit.

Intensity of the electron beam is controlled by setting voltage levels on the

## Chapter 2

overview of Graphics  Systems ating voltage can be generated with a positively charged metal coating on the inside of  the CRT envelope near the phosphor screen, or an accelerating anode can be used, as in Fig. 2-3. Sometimes the electron gun is built to contain the accelerating anode and focusing system within the same unit.

Intensity of  the electron  beam is controlled  by  setting voltage levels on the control grid, which is a metal cylinder that fits over the cathode. A high negative voltage applied to the control grid  will shut off  the beam  by repelling eledrons and stopping them from passing through the small hole at the end of the control grid  structure. A smaller negative voltage on the control grid simply decreases the number of  electrons passing through. Since the amount of  light emitted by the phosphor coating depends on the number of electrons striking the screen, we control the brightness of  a display by varying the voltage on the control grid. We specify  the intensity level for individual screen positions with graphics software commands, as discussed in Chapter 3.

The focusing system in a CRT is needed  to force the electron beam to converge into a small spot as it strikes the phosphor. Otherwise, the electrons would repel each other, and the beam would spread out as it approaches the screen. Focusing is accomplished with either electric or magnetic fields. Electrostatic focusing is commonly used in television and computer graphics monitors. With electrostatic focusing,  the elwtron beam  passes  through a positively  charged  metal cylinder that  forms an electrostatic lens, as shown in  Fig. 2-3. The action of   the electrostatic lens fdcuses the electron  beam at the center of  the screen, in exactly the same way that an optical lens focuses a beam of  hght at a particular  focal distance. Similar lens focusing effects can be accomplished with a magnetic field set up by a coil mounted around the outside of  the CRT envelope. Magnetic lens f c cusing  produces  the  smallest  spot  size  on  the  screen  and  is  used  in  specialpurpose devices.

Additional focusing hardware is used in high-precision  systems to keep the beam in focus at all m n positions.  The distance that  thc electron beam must travel to different  points on the screen varies because thc radius of  curvature for most  CRTs is greater than  the distance  from  the focusing system  to  the  screen center. Therefore, the electron beam will be focused properly only at the center ot the screen.  As the beam moves to the outer edges of  the screen, displayed images become blurred. To compensate for this, the system can  adjust  the focusing according to the screen position of the beam.

As with focusing,  deflection  of  the electron  beam  can  be  controlled  either with electric fields or with magnetic fields. Cathode-ray tubes are now commonl!. constructed  with  magnetic deflection coils mounted  on  the outside of  the CRT envelope,  as illustrated  in Fig. 2-2. Two pairs of  coils are used,  with  the coils in each pair mounted on opposite sides of  the neck of  the CRT envelope. One pair is mounted  on the top and bottom of  the neck, and the other pair  is mounted  on opposite sides of  the neck. The magnetic,  field produced by each pair of  coils results in a transverse deflection force that is perpendicular both to the direction of the magnetic field and to the direction of  travel of  the electron beam. Horizontal deflection is accomplished  with one pair of  coils, and  vertical  deflection  by  the other pair. The proper deflection  amounts are attained  by adjusting the current through  the  coils.  When  electrostatic  deflection  is  used,  two  pairs  of parallel plates are mounted  inside the CRT envelope. One pair oi plates is mounted  horizontally to control the vertical deflection, and the other pair is mounted verticall!. to control horizontal deflection  (Fig. 2-4).

Spots of  light  are produced  on the screen by  the transfer  of  the CRT beam energy  to the  phosphor. When  the electrons in the beam collide with  the phos-

Base

Connector

Pins

Figure 2-4

System

Vertical

Plates

Phospinor-

Coated

Screen

Figure 2-4 Electmstatic  deflection o f   the electron  beam in a CRT.

<!-- image -->

quantums of light energy. What we see on the screen is the combined effect of all phor coating, they are stopped and thek kinetic energy is absorbed by the phosphor. Part of   the beam energy is converted by  friction into heat energy, and the remainder causes electrons in the phosphor atoms to move  up to higher quantum-energy  levels. After  a short  time,  the  "excited  phosphor  electrons begin dropping back to their stable ground state, giving up their extra energy as small quantums of  Light energy.  What we see on the screen is the combined effect o f   all the electron light emissions: a glowing spot that quickly fades after all the excited phosphor electrons have returned to their ground energy level.  The frequency (or color) of  the light emitted by  the phosphor is proportional to the energy difference between the excited quantum state and the ground state.

Different hnds of  phosphors are available for use in a CRT. Besides color, a mapr difference  between phosphors is their persistence: how long they continue to emit light (that is, have excited electrons returning to the ground state) after the CRT beam is removed. Persistence is defined as the time it takes the emitted light  from  the  screen  to  decay  to  one-tenth  of its  original  intensity.  Lowerpersistence phosphors  require higher refresh rates to maintain a picture on the screen without flicker. A phosphor with low persistence  is use f u l for animation; a high-persistence phosphor  is use f u l for  displaying  highly  complex, static pictures. Although some phosphors have a persistence greater than 1  second, graphics monitors are usually constructed with a persistence in the range from 10 to 60 microseconds.

Figure 2-5 shows the intensity distribution of  a spot on the screen. The intensity is greatest at the center of t h e spot, and decreaws with a Gaussian distribution  out  to  the  edges o f   the spot. This  distribution corresponds to the msssectional  electron density distribution o f   the CRT beam. '

The maximum number of points that can be displayed without overlap on a CRT is referred to as the resolution. A more precise definition o f   m!ution  is the number of  points per centimeter that can be plotted horizontally and vertically, although it is often simply stated as the total number o f   points in each direction. Spot intensity has a Gaussian distribution (Fig. 2-5), so two adjacent spok will appear distinct as long as their separation is greater than the diameter at which each spot  has an intensity of  about 60 percent of  that at the center of  the spot. This overlap position is illustrated in Fig. 2-6. Spot size also  depends on intensity. As  more  electrons are accelerated  toward  the phospher  per  second,  the CRT beam diameter and the illuminated spot increase. In addition, the increased excitation energy tends to spread to neighboring phosphor atoms not directly in the

Focusing

Deflection

Fipn 2-5 Intensity distribution  o f   an illuminated phosphor spot o n a CRT screen.

<!-- image -->

39

40

Chapter 2

Overview of Graphics Systems path of the beam, which further increases the spot diameter. Thus, resolution of a

focusing and deflection systems. Typical resolution on high-quality systems is

CRT is dependent on the type of phosphor, the intensity to be displayed, and the

1280 by 1024, with higher resolutions available on many systems. High- resolution systems are often referred to as high-definition systems. The physical

Chrpcer

2

Overview of Graphics Sptems

Figure 2-6

Two illuminated phosphor when their separation is

spots are distinguishable greater than the diameter at

<!-- image -->

which a spot intensity has maximum.

Figure 2-6 Two illuminated phosphor spots are distinguishable when their separation is greater than the diameter at which a spot intensity has fallen to 60 percent o f maximum.

path of  the beam, which further increases the spot diameter. Thus, resolution of  a CRT is dependent on the type of  phosphor, the intensity to be displayed, and the focusing and deflection systems. Typical  resolution  on high-quality  systems  is 1280 by 1024, with  higher  resolutions available  on many  systems. Highresolution  systems are often  referred  to as high-definition systems. The physical size of a graphics monitor is given as the length of  the screen diagonal, with sizes varying  from about 12 inches to 27 inches or more. A CRT monitor can be  attached  to a variety o f   computer systems, so the number of  screen points that can actually be plotted  depends on the capabilities of  the system  to which  it  is attached.

Another property of  video monitors is aspect ratio. This number gives the ratio  o f vertical  points  to  horizontal  points  necessary  to  produce equal-length lines in both directions on the screen. (Sometimes aspect ratio is stated in terms o f the ratio of horizontal to vertical points.) An aspect ratio of 3/4 means that a vertical line plotted  with three points has the same length as a horizontal  line plotted with four points.

electron beam moves across each row, the beam intensity is turned on and off to

## Raster-Scan Displays

The most common type of  graphics monitor employing a CRT is the raster-scan display,  based  on television  technology.  In  a  raster-scan  system,  the  electron beam is swept across the screen, one row at a time from top to bottom.  As  the eledron beam moves across each row, the beam intensity is turned on and off  to create  a  pattern  of  illuminated  spots.  Picture  definition  is stored  in a  memory area called the refresh buffer or frame buffer. This memory area holds the set of intensity  values  for  all  the  screen  points.  Stored  intensity  values  are  then  retrieved from the refresh buffer and "painted"  on the screen one row (scan line) at a  time  (Fig. 2-7). Each screen point  is  referred  to  as  a  pixel  or pel  (shortened fonns of  picture element). The capability of  a  raster-scan  system  to store intensity information for each screen point makes it well suited for the realistic displav of  scenes containing subtle shading and color patterns.  Home television sets and printers are examples of  other systems using raster-scan  methods.

intensity  range for pixel  positions  depends on  the capability of  the raster system. In a simple black-and-white system, each screen point is either on or off, so only one  bit per pixel is needed  to control the intensity of  screen positions. For a bilevel  system, a bit value of  1 indicates that the electron beam is to be t u r n 4 on at that position, and a value of 0 indicates that the beam intensity is to be off. Additional bits are needed  when color and intensity variations can be displayed. Up to 24 bits per pixel are included  in high-quality  systems, which can require severaI megabytes of  storage for the frame buffer, depending on the resolution of the system. A system with 24 bits  per pixel  and a  screen resolution  of  1024 bv 1024 requires 3 megabytes of  storage for the frame buffer. On a black-and-white system with one bit per pixeI, the frame buffer is commonly called a bitmap. For systems with  multiple  bits per pixel,  the frame buffer  is Aten referred  to as a pixmap.

Refreshing  on  raster-scan  displays is  carried  out  at  the  rate  of  60  to  80 frames per second, although some systems are designed for higher refresh rates. Sometimes, refresh  rates  are described  in  units of  cycles  per  second, or Hertz (Hz), where a cycle corresponds  to one frame. Using  these units, we would describe a refresh rate of 60 frames per second as simply 60 Hz. At the end of  each scan line, the electron beam returns to the left side o f   the screen to begin displaving the next scan line. The return  to the left of  the screen, after  refreshing each pixmap.

Figure 2-7

each scan line.

scan line, is called the horizontal retrace of the electron beam. And at the end of

Figure 2-7

<!-- image -->

A raster-scan system displays an object as a set o f   dismte  points across each scan line.

two passes using an interlaced refresh procedure. In the first pass, the beam sweeps across every other scan line from top to bottom. Then after the vertical re-

scan line, is called the horizontal retrace o f   the electron beam. And at the end of each frame (displayed in 1/80th to 1/60th of   a second),  the electron beam returns (vertical retrace) to the top left comer o f   the screen to begin the next frame.

On some raster-scan systems (and in TV sets), each  frame is  displayed in two  passes  using  an  interlaced  refresh  pmedure.  In  the  first  pass,  the  beam sweeps across every other scan line fmm top to bottom. Then after the vertical retrace, the beam sweeps out the remaining scan lines (Fig. 2-8). Interlacing o f   the scan lines in this way allows us to see the entire s m n   displayed in one-half  the time it would have taken to sweep a m s s   all the lines at once fmm top to bottom. Interlacing is primarily used with slower refreshing rates. On an older, 30 frameper-second, noninterlaced display, for  instance, some  flicker is  noticeable.  But with interlacing, each o f   the two passes can be accomplished in 1/60th o f   a second, which brings the refresh rate nearer to 60 frames per second. This is an effective technique for avoiding flicker, providing that adjacent scan lines contain similar display information.

## Random-Scan Displays

When operated as a random-scan display unit, a CRT has the electron beam directed  only to the parts of  the screen where a picture is to be drawn.  Randomscan monitors draw a picture one line at a time and for this reason are also referred  to  as  vector displays (or stroke-writing  or  calligraphic  diisplays). The component lines of  a picture can be drawn and refreshed by a random-scan sys-

41