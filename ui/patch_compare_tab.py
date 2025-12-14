import sys
import os
from xml.etree import ElementTree as ET

NEW_LAYOUT_XML = r"""
<wrapper>
  <item>
    <layout class="QHBoxLayout" name="comparePoseRoot">
      <!-- Left side: two video panes -->
      <item>
        <layout class="QHBoxLayout" name="videosRow">
          <!-- Teacher video -->
          <item>
            <widget class="QGroupBox" name="groupTeacher">
              <property name="title"><string>Reference (Teacher)</string></property>
              <layout class="QVBoxLayout" name="groupTeacherCol">
                <item>
                  <widget class="QLabel" name="videoTeacher">
                    <property name="minimumSize"><size><width>460</width><height>260</height></size></property>
                    <property name="sizePolicy">
                      <sizepolicy hsizetype="Expanding" vsizetype="Expanding">
                        <horstretch>0</horstretch><verstretch>0</verstretch>
                      </sizepolicy>
                    </property>
                    <property name="frameShape"><enum>QFrame::StyledPanel</enum></property>
                    <property name="alignment"><set>Qt::AlignCenter</set></property>
                    <property name="text"><string>Teacher Video</string></property>
                  </widget>
                </item>
                <item>
                  <layout class="QHBoxLayout" name="groupTeacherButtons">
                    <item>
                      <widget class="QPushButton" name="btnSelectTeacherVideo">
                        <property name="text"><string>Select Reference Video…</string></property>
                      </widget>
                    </item>
                    <item>
                      <widget class="QLabel" name="lblTeacherFilename">
                        <property name="text"><string>No file selected</string></property>
                        <property name="toolTip"><string>Selected reference video file</string></property>
                      </widget>
                    </item>
                  </layout>
                </item>
              </layout>
            </widget>
          </item>

          <!-- Student video -->
          <item>
            <widget class="QGroupBox" name="groupStudent">
              <property name="title"><string>Target (Student)</string></property>
              <layout class="QVBoxLayout" name="groupStudentCol">
                <item>
                  <widget class="QLabel" name="videoStudent">
                    <property name="minimumSize"><size><width>460</width><height>260</height></size></property>
                    <property name="sizePolicy">
                      <sizepolicy hsizetype="Expanding" vsizetype="Expanding">
                        <horstretch>0</horstretch><verstretch>0</verstretch>
                      </sizepolicy>
                    </property>
                    <property name="frameShape"><enum>QFrame::StyledPanel</enum></property>
                    <property name="alignment"><set>Qt::AlignCenter</set></property>
                    <property name="text"><string>Student Video</string></property>
                  </widget>
                </item>
                <item>
                  <layout class="QHBoxLayout" name="groupStudentButtons">
                    <item>
                      <widget class="QPushButton" name="btnSelectStudentVideo">
                        <property name="text"><string>Select Target Video…</string></property>
                      </widget>
                    </item>
                    <item>
                      <widget class="QLabel" name="lblStudentFilename">
                        <property name="text"><string>No file selected</string></property>
                        <property name="toolTip"><string>Selected target video file</string></property>
                      </widget>
                    </item>
                  </layout>
                </item>
              </layout>
            </widget>
          </item>
        </layout>
      </item>

      <!-- Right panel: score + error hints -->
      <item>
        <widget class="QFrame" name="panelRight">
          <property name="minimumSize"><size><width>240</width><height>0</height></size></property>
          <property name="frameShape"><enum>QFrame::StyledPanel</enum></property>
          <layout class="QVBoxLayout" name="rightCol">
            <!-- Score -->
            <item>
              <widget class="QGroupBox" name="groupScore">
                <property name="title"><string>Score</string></property>
                <layout class="QVBoxLayout" name="scoreCol">
                  <item>
                    <widget class="QLCDNumber" name="lcdScore">
                      <property name="segmentStyle"><enum>QLCDNumber::Flat</enum></property>
                      <property name="digitCount"><number>4</number></property>
                    </widget>
                  </item>
                  <item>
                    <widget class="QLabel" name="lblScoreHint">
                      <property name="text"><string>Higher is better</string></property>
                      <property name="alignment"><set>Qt::AlignCenter</set></property>
                    </widget>
                  </item>
                </layout>
              </widget>
            </item>

            <!-- Error hints -->
            <item>
              <widget class="QGroupBox" name="groupErrors">
                <property name="title"><string>Error Hints</string></property>
                <layout class="QVBoxLayout" name="errorsCol">
                  <item>
                    <widget class="QListWidget" name="listErrors">
                      <property name="toolTip"><string>Detected mistakes and suggestions</string></property>
                    </widget>
                  </item>
                  <item>
                    <widget class="QLabel" name="lblErrorNote">
                      <property name="text"><string>Example: Keep your wrist aligned.</string></property>
                      <property name="wordWrap"><bool>true</bool></property>
                    </widget>
                  </item>
                </layout>
              </widget>
            </item>

            <!-- Actions -->
            <item>
              <layout class="QHBoxLayout" name="actionsRow">
                <item>
                  <widget class="QPushButton" name="btnStartAnalysis">
                    <property name="text"><string>Start Analysis</string></property>
                  </widget>
                </item>
                <item>
                  <widget class="QPushButton" name="btnStopAnalysis">
                    <property name="text"><string>Stop</string></property>
                  </widget>
                </item>
              </layout>
            </item>
          </layout>
        </widget>
      </item>
    </layout>
  </item>
</wrapper>
"""

def find_widget_by_name(root, name):
    for w in root.iter('widget'):
        if w.attrib.get('name') == name:
            return w
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_compare_tab.py app.ui")
        return
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    ET.register_namespace('', "http://www.qt-project.org/ui")  # safe even if not used
    tree = ET.parse(path)
    root = tree.getroot()

    tab = find_widget_by_name(root, 'tabPoseCompare')
    if tab is None:
        print("tabPoseCompare not found. Please ensure your tab name is 'tabPoseCompare'.")
        return

    layout = None
    for l in tab.findall('layout'):
        if l.attrib.get('name') == 'layoutPoseCompare':
            layout = l
            break
    if layout is None:
        print("layoutPoseCompare not found inside tabPoseCompare.")
        return

    # Clear current children of layoutPoseCompare
    for child in list(layout):
        layout.remove(child)

    # Parse new content and append children into layoutPoseCompare
    wrapper = ET.fromstring(NEW_LAYOUT_XML)
    for child in list(wrapper):
        # child is <item> ... we append directly under layoutPoseCompare
        layout.append(child)

    # Write backup and patched file
    backup = path + ".bak"
    if not os.path.exists(backup):
        with open(backup, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)

    out_path = path  # overwrite original
    with open(out_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    print("Patched successfully!")
    print(f"- Backup: {backup}")
    print(f"- Updated: {out_path}")

if __name__ == "__main__":
    main()
